// mobile/AboutScreen.tsx
// About screen: displays app info and model details
import React from "react";
import { ScrollView, Text, StyleSheet } from "react-native";
import { APP_CONFIG } from "./config";

export default function AboutScreen() {
  return (
    <ScrollView style={styles.container}>
      <Text style={styles.content}>{APP_CONFIG.aboutText}</Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#1a1a1a",
    padding: 20,
  },
  content: {
    fontSize: 16,
    color: "#cccccc",
    lineHeight: 24,
  },
});
