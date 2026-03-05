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
  Alert,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
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
        Alert.alert("Permission Required", "Camera permission is needed.");
        return;
      }
    } else {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== "granted") {
        Alert.alert("Permission Required", "Photo library permission is needed.");
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
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
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
      </View>

      <View style={styles.imageArea}>
        {imageUri ? (
          <Image source={{ uri: imageUri }} style={styles.preview} />
        ) : (
          <Text style={styles.placeholder}>Select an image to classify</Text>
        )}
        {loading && <ActivityIndicator size="large" color="#4CAF50" style={styles.loader} />}
      </View>

      {error && (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText} numberOfLines={2}>{error}</Text>
        </View>
      )}

      {result && (
        <View style={styles.resultContainer}>
          <View style={styles.resultHeader}>
            <Text style={styles.resultTitle}>Predictions</Text>
            <Text style={styles.inferenceTime}>{result.inference_time_ms} ms</Text>
          </View>
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
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#1a1a1a",
    paddingHorizontal: 16,
  },
  header: {
    alignItems: "center",
    paddingTop: 8,
  },
  title: {
    fontSize: 26,
    fontWeight: "bold",
    color: "#ffffff",
  },
  status: {
    fontSize: 13,
    color: "#aaaaaa",
    marginVertical: 4,
  },
  buttonRow: {
    flexDirection: "row",
    gap: 12,
    marginVertical: 8,
  },
  button: {
    backgroundColor: "#333333",
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 50,
    borderWidth: 2,
    borderColor: "#555555",
  },
  buttonDisabled: {
    opacity: 0.4,
  },
  buttonText: {
    fontSize: 15,
    color: "#ffffff",
  },
  imageArea: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  preview: {
    width: "85%",
    aspectRatio: 1,
    borderRadius: 12,
  },
  placeholder: {
    color: "#555555",
    fontSize: 16,
  },
  loader: {
    position: "absolute",
  },
  errorContainer: {
    backgroundColor: "#4a1a1a",
    borderRadius: 8,
    padding: 8,
    marginBottom: 4,
  },
  errorText: {
    color: "#ff6b6b",
    fontSize: 12,
  },
  resultContainer: {
    backgroundColor: "#2a2a2a",
    borderRadius: 12,
    padding: 12,
    marginBottom: 8,
  },
  resultHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 6,
  },
  resultTitle: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#ffffff",
  },
  inferenceTime: {
    fontSize: 12,
    color: "#888888",
  },
  predictionRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 5,
    borderBottomWidth: 1,
    borderBottomColor: "#333333",
  },
  className: {
    fontSize: 14,
    color: "#ffffff",
  },
  probability: {
    fontSize: 14,
    color: "#4CAF50",
    fontWeight: "bold",
  },
});
