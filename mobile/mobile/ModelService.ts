// mobile/ModelService.ts
// Core inference logic: loads TF.js model from bundled assets, classifies images
import * as tf from "@tensorflow/tfjs";
import { bundleResourceIO, decodeJpeg } from "@tensorflow/tfjs-react-native";
import { APP_CONFIG } from "./config";

const modelJSON = require("./assets/model_tfjs/model.json");
const modelWeights = require("./assets/model_tfjs/group1-shard1of1.bin");
const classesData: string[] = require("./assets/model_tfjs/classes.json");

export interface Prediction {
  class: string;
  probability: number;
}

export interface ClassificationResult {
  predictions: Prediction[];
  inference_time_ms: number;
}

let model: tf.GraphModel | null = null;

export async function initializeTF(): Promise<void> {
  await tf.ready();
}

export async function loadModel(): Promise<void> {
  if (model) return;
  model = await tf.loadGraphModel(bundleResourceIO(modelJSON, modelWeights));
  // Warm up with a dummy prediction
  const warmup = tf.zeros([1, APP_CONFIG.imageSize, APP_CONFIG.imageSize, 3]);
  model.predict(warmup);
  warmup.dispose();
}

export function isModelReady(): boolean {
  return model !== null;
}

export function getClasses(): string[] {
  return classesData;
}

export async function classifyImage(
  imageUri: string
): Promise<ClassificationResult> {
  if (!model) throw new Error("Model not loaded");

  // Read image as raw bytes using fetch (avoids deprecated expo-file-system)
  const response = await fetch(imageUri);
  const arrayBuffer = await response.arrayBuffer();
  const rawImageTensor = decodeJpeg(new Uint8Array(arrayBuffer));

  const start = performance.now();

  const processedTensor = tf.tidy(() => {
    const resized = tf.image.resizeBilinear(rawImageTensor, [
      APP_CONFIG.imageSize,
      APP_CONFIG.imageSize,
    ]);
    const normalized = resized.toFloat().div(127.5).sub(1.0);
    return normalized.expandDims(0);
  });

  const predictionTensor = model.predict(processedTensor) as tf.Tensor;
  const probabilities = await predictionTensor.data();
  const inferenceTime = performance.now() - start;

  rawImageTensor.dispose();
  processedTensor.dispose();
  predictionTensor.dispose();

  const predictions = classesData
    .map((cls, i) => ({ class: cls, probability: probabilities[i] }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, APP_CONFIG.numPredictions);

  return {
    predictions,
    inference_time_ms: Math.round(inferenceTime * 100) / 100,
  };
}
