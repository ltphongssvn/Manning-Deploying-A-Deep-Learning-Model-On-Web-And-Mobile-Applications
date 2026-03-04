// mobile/HomeScreen.tsx
// Main screen: image selection (camera/gallery), model loading, prediction display
import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  Image,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  ScrollView,
  Alert,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { initializeTF, loadModel, isModelReady, classifyImage } from "./ModelService";
import type { ClassificationResult } from "./ModelService";
import { APP_CONFIG } from "./config";

export default function HomeScreen() {
  const [modelStatus, setModelStatus] = useState<string>("Loading TF...");
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        setModelStatus("Initializing TF.js...");
        await initializeTF();
        setModelStatus("Loading model...");
        await loadModel();
        setModelStatus("Ready ✅");
      } catch (e) {
        setModelStatus(`Failed: ${e}`);
        console.error("Model load error:", e);
      }
    })();
  }, []);

  const pickImage = async (useCamera: boolean) => {
    setError(null);

    if (useCamera) {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== "granted") {
        Alert.alert(
          "Permission Required",
          "Camera permission is needed to take photos."
        );
        return;
      }
    } else {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== "granted") {
        Alert.alert(
          "Permission Required",
          "Photo library permission is needed to select images."
        );
        return;
      }
    }

    const pickerFn = useCamera
      ? ImagePicker.launchCameraAsync
      : ImagePicker.launchImageLibraryAsync;

    try {
      const pickerResult = await pickerFn({
        mediaTypes: ["images"],
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
      });

      if (!pickerResult.canceled && pickerResult.assets[0]) {
        const uri = pickerResult.assets[0].uri;
        setImageUri(uri);
        setResult(null);
        await runPrediction(uri);
      }
    } catch (e) {
      console.error("Image picker error:", e);
      setError(`Failed to pick image: ${e}`);
    }
  };

  const runPrediction = async (uri: string) => {
    if (!isModelReady()) return;
    setLoading(true);
    setError(null);
    try {
      const classificationResult = await classifyImage(uri);
      setResult(classificationResult);
    } catch (e) {
      console.error("Prediction error:", e);
      setError(`Prediction failed: ${e}`);
    }
    setLoading(false);
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>{APP_CONFIG.title}</Text>
      <Text style={styles.status}>Model Status: {modelStatus}</Text>

      <View style={styles.buttonRow}>
        <TouchableOpacity
          style={[styles.button, !isModelReady() && styles.buttonDisabled]}
          onPress={() => pickImage(true)}
          disabled={!isModelReady()}
        >
          <Text style={styles.buttonText}>📷 Camera</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.button, !isModelReady() && styles.buttonDisabled]}
          onPress={() => pickImage(false)}
          disabled={!isModelReady()}
        >
          <Text style={styles.buttonText}>🖼️ Gallery</Text>
        </TouchableOpacity>
      </View>

      {imageUri && (
        <Image source={{ uri: imageUri }} style={styles.preview} />
      )}

      {loading && <ActivityIndicator size="large" color="#4CAF50" />}

      {error && (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      {result && (
        <View style={styles.resultContainer}>
          <Text style={styles.resultTitle}>Predictions</Text>
          <Text style={styles.inferenceTime}>
            Inference: {result.inference_time_ms} ms
          </Text>
          {result.predictions.map((p) => (
            <View key={p.class} style={styles.predictionRow}>
              <Text style={styles.className}>{p.class}</Text>
              <Text style={styles.probability}>
                {(p.probability * 100).toFixed(APP_CONFIG.probabilityPrecision)}%
              </Text>
            </View>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    alignItems: "center",
    padding: 20,
    backgroundColor: "#1a1a1a",
  },
  title: {
    fontSize: 32,
    fontWeight: "bold",
    color: "#ffffff",
    marginTop: 40,
    marginBottom: 8,
  },
  status: {
    fontSize: 16,
    color: "#aaaaaa",
    marginBottom: 20,
  },
  buttonRow: {
    flexDirection: "row",
    gap: 16,
    marginBottom: 20,
  },
  button: {
    backgroundColor: "#333333",
    paddingVertical: 14,
    paddingHorizontal: 24,
    borderRadius: 50,
    borderWidth: 2,
    borderColor: "#555555",
  },
  buttonDisabled: {
    opacity: 0.4,
  },
  buttonText: {
    fontSize: 18,
    color: "#ffffff",
  },
  preview: {
    width: 300,
    height: 300,
    borderRadius: 12,
    marginBottom: 20,
  },
  errorContainer: {
    backgroundColor: "#4a1a1a",
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
    width: "100%",
  },
  errorText: {
    color: "#ff6b6b",
    fontSize: 14,
  },
  resultContainer: {
    width: "100%",
    backgroundColor: "#2a2a2a",
    borderRadius: 12,
    padding: 16,
    marginTop: 10,
  },
  resultTitle: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#ffffff",
    marginBottom: 8,
  },
  inferenceTime: {
    fontSize: 14,
    color: "#888888",
    marginBottom: 12,
  },
  predictionRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: "#333333",
  },
  className: {
    fontSize: 16,
    color: "#ffffff",
  },
  probability: {
    fontSize: 16,
    color: "#4CAF50",
    fontWeight: "bold",
  },
});
