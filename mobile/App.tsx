// mobile/App.tsx
// Root component: bottom tab navigation (Home + About)
import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { StatusBar } from "expo-status-bar";
import HomeScreen from "./HomeScreen";
import AboutScreen from "./AboutScreen";

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <>
      <StatusBar style="light" />
      <NavigationContainer>
        <Tab.Navigator
          screenOptions={{
            headerStyle: { backgroundColor: "#1a1a1a" },
            headerTintColor: "#ffffff",
            tabBarStyle: { backgroundColor: "#1a1a1a", borderTopColor: "#333" },
            tabBarActiveTintColor: "#4CAF50",
            tabBarInactiveTintColor: "#888888",
          }}
        >
          <Tab.Screen
            name="Home"
            component={HomeScreen}
            options={{ tabBarLabel: "Home", tabBarIcon: () => <></> }}
          />
          <Tab.Screen
            name="About"
            component={AboutScreen}
            options={{ tabBarLabel: "About", tabBarIcon: () => <></> }}
          />
        </Tab.Navigator>
      </NavigationContainer>
    </>
  );
}
