import { useState, useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface Prediction {
  class: string;
  probability: number;
}

interface InferenceResult {
  predictions: Prediction[];
  inference_time_ms: number;
}

function App() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState("");
  const [preview, setPreview] = useState<string | null>(null);
  const [serverResult, setServerResult] = useState<InferenceResult | null>(null);
  const [browserResult, setBrowserResult] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState({ server: false, browser: false });
  const [tfModel, setTfModel] = useState<tf.GraphModel | null>(null);
  const [modelStatus, setModelStatus] = useState("Not loaded");
  const [classes, setClasses] = useState<string[]>([]);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    fetch(`${API_URL}/api/classes`)
      .then((r) => r.json())
      .then((data) => setClasses(data.classes))
      .catch(() => setClasses(["apple_pie", "caesar_salad", "falafel"]));
  }, []);

  const loadBrowserModel = async () => {
    if (tfModel) return;
    setModelStatus("Loading...");
    try {
      const model = await tf.loadGraphModel(
        `${API_URL}/artifacts/model_tfjs/model.json`
      );
      setTfModel(model);
      setModelStatus("Ready");
    } catch (e) {
      setModelStatus("Failed to load");
      console.error("TF.js model load error:", e);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      setImageUrl("");
      setPreview(URL.createObjectURL(file));
      setServerResult(null);
      setBrowserResult(null);
    }
  };

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setImageUrl(e.target.value);
    setImageFile(null);
    setPreview(e.target.value);
    setServerResult(null);
    setBrowserResult(null);
  };

  const handleClear = () => {
    setImageFile(null);
    setImageUrl("");
    setPreview(null);
    setServerResult(null);
    setBrowserResult(null);
  };

  const predictServer = async () => {
    setLoading((p) => ({ ...p, server: true }));
    setServerResult(null);
    try {
      let res;
      if (imageFile) {
        const formData = new FormData();
        formData.append("file", imageFile);
        res = await fetch(`${API_URL}/api/predict`, {
          method: "POST",
          body: formData,
        });
      } else if (imageUrl) {
        res = await fetch(`${API_URL}/api/predict_url`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: imageUrl }),
        });
      }
      if (res) {
        const data = await res.json();
        setServerResult(data);
      }
    } catch (e) {
      console.error("Server prediction error:", e);
    }
    setLoading((p) => ({ ...p, server: false }));
  };

  const predictBrowser = async () => {
    if (!tfModel || !imgRef.current) return;
    setLoading((p) => ({ ...p, browser: true }));
    setBrowserResult(null);
    try {
      const start = performance.now();
      const tensor = tf.tidy(() => {
        const img = tf.browser
          .fromPixels(imgRef.current!)
          .resizeBilinear([224, 224])
          .toFloat();
        const normalized = img.div(127.5).sub(1.0);
        return normalized.expandDims(0);
      });
      const predictions = (await tfModel.predict(tensor)) as tf.Tensor;
      const probs = await predictions.data();
      const inferenceTime = performance.now() - start;
      tensor.dispose();
      predictions.dispose();

      const results = classes
        .map((c, i) => ({ class: c, probability: probs[i] }))
        .sort((a, b) => b.probability - a.probability);

      setBrowserResult({
        predictions: results,
        inference_time_ms: Math.round(inferenceTime * 100) / 100,
      });
    } catch (e) {
      console.error("Browser prediction error:", e);
    }
    setLoading((p) => ({ ...p, browser: false }));
  };

  const handlePredict = () => {
    predictServer();
    if (tfModel && imgRef.current) predictBrowser();
  };

  const ResultTable = ({
    title,
    result,
    isLoading,
  }: {
    title: string;
    result: InferenceResult | null;
    isLoading: boolean;
  }) => (
    <div style={{ flex: 1, minWidth: 250 }}>
      <h3>{title}</h3>
      {isLoading && <p>Processing...</p>}
      {result && (
        <>
          <p style={{ fontSize: "0.85em", color: "#666" }}>
            Inference time: {result.inference_time_ms} ms
          </p>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9em" }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left", padding: 6, borderBottom: "2px solid #ddd" }}>Class</th>
                <th style={{ textAlign: "right", padding: 6, borderBottom: "2px solid #ddd" }}>Probability</th>
              </tr>
            </thead>
            <tbody>
              {result.predictions.map((p) => (
                <tr key={p.class}>
                  <td style={{ padding: 6, borderBottom: "1px solid #eee" }}>{p.class}</td>
                  <td style={{ padding: 6, borderBottom: "1px solid #eee", textAlign: "right" }}>
                    {(p.probability * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: 20, fontFamily: "sans-serif" }}>
      <h1>Classify Food Image Using Deep Learning</h1>

      <div style={{ marginBottom: 20 }}>
        <h3>Provide a URL</h3>
        <input
          type="text"
          value={imageUrl}
          onChange={handleUrlChange}
          placeholder="Enter image URL..."
          style={{ width: "100%", padding: 8, fontSize: 14 }}
        />
      </div>

      <div style={{ marginBottom: 20 }}>
        <h3>OR Upload an image</h3>
        <input type="file" accept="image/*" onChange={handleFileChange} />
      </div>

      {preview && (
        <div style={{ marginBottom: 20 }}>
          <img
            ref={imgRef}
            src={preview}
            alt="Preview"
            crossOrigin="anonymous"
            style={{ maxWidth: 400, maxHeight: 400, borderRadius: 8 }}
          />
        </div>
      )}

      <div style={{ display: "flex", gap: 10, marginBottom: 20, alignItems: "center" }}>
        <button
          onClick={handlePredict}
          disabled={!preview || loading.server}
          style={{ padding: "8px 20px", fontSize: 16, background: "#4CAF50", color: "white", border: "none", borderRadius: 4, cursor: "pointer" }}
        >
          Predict
        </button>
        <button
          onClick={handleClear}
          style={{ padding: "8px 20px", fontSize: 16, background: "#2196F3", color: "white", border: "none", borderRadius: 4, cursor: "pointer" }}
        >
          Clear
        </button>
        <button
          onClick={loadBrowserModel}
          disabled={!!tfModel}
          style={{ padding: "8px 20px", fontSize: 16, background: "#FF9800", color: "white", border: "none", borderRadius: 4, cursor: "pointer" }}
        >
          Load Browser Model ({modelStatus})
        </button>
      </div>

      <div style={{ display: "flex", gap: 40, flexWrap: "wrap" }}>
        <ResultTable title="Server Side Inference" result={serverResult} isLoading={loading.server} />
        <ResultTable title="Client Side Inference" result={browserResult} isLoading={loading.browser} />
      </div>

      <div style={{ marginTop: 40, fontSize: "0.85em", color: "#666" }}>
        <h3>Notes</h3>
        <p>Server side inference sends the image to the backend API for prediction.</p>
        <p>Browser side inference loads the TF.js model locally and runs prediction entirely in the browser.</p>
      </div>
    </div>
  );
}

export default App;
