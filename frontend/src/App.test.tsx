// frontend/src/App.test.tsx
/**
 * Component tests for App.tsx Food Classifier UI.
 *
 * TensorFlow.js is mocked to avoid WebGL/WASM dependencies in jsdom.
 * fetch is mocked to avoid real API calls (Fast, Independent tests).
 *
 * Tests follow Given/When/Then structure and ZOMBIES mnemonic:
 * - Simple: renders core UI elements
 * - Interfaces: button states, input contracts
 * - Boundaries: disabled states, empty inputs
 * - Zero: initial render with no image selected
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import App from "./App";

// Mock TensorFlow.js — jsdom has no WebGL
vi.mock("@tensorflow/tfjs", () => ({
  loadGraphModel: vi.fn(),
  tidy: vi.fn(),
  browser: { fromPixels: vi.fn() },
}));

// Mock fetch globally
const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

beforeEach(() => {
  vi.clearAllMocks();
  // Default: /api/classes returns our 3 food classes
  mockFetch.mockResolvedValue({
    ok: true,
    json: async () => ({ classes: ["apple_pie", "caesar_salad", "falafel"] }),
  });
});

/**
 * Helper: render App and wait for initial useEffect (fetch /api/classes)
 * to complete, preventing act(...) warnings from async state updates.
 */
async function renderApp() {
  render(<App />);
  await waitFor(() => {
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/classes")
    );
  });
}

// ---------------------------------------------------------------------------
// Simple: Core UI elements render
// ---------------------------------------------------------------------------
describe("App renders core UI elements", () => {
  it("displays the main heading", async () => {
    // Given: the app is rendered and settled
    await renderApp();
    // Then: heading is visible
    expect(
      screen.getByRole("heading", { name: /classify food image/i })
    ).toBeInTheDocument();
  });

  it("displays URL input field", async () => {
    // Given: the app is rendered and settled
    await renderApp();
    // Then: URL input is present with placeholder
    expect(screen.getByPlaceholderText(/enter image url/i)).toBeInTheDocument();
  });

  it("displays file upload input", async () => {
    // Given: the app is rendered and settled
    await renderApp();
    // Then: file input exists
    const fileInput = document.querySelector('input[type="file"]');
    expect(fileInput).toBeInTheDocument();
  });

  it("displays Predict button", async () => {
    // Given: the app is rendered and settled
    await renderApp();
    // Then: Predict button exists
    expect(screen.getByRole("button", { name: /predict/i })).toBeInTheDocument();
  });

  it("displays Clear button", async () => {
    // Given: the app is rendered and settled
    await renderApp();
    // Then: Clear button exists
    expect(screen.getByRole("button", { name: /clear/i })).toBeInTheDocument();
  });

  it("displays Load Browser Model button", async () => {
    // Given: the app is rendered and settled
    await renderApp();
    // Then: Load Browser Model button exists
    expect(
      screen.getByRole("button", { name: /load browser model/i })
    ).toBeInTheDocument();
  });

  it("displays Server and Client inference headings", async () => {
    // Given: the app is rendered and settled
    await renderApp();
    // Then: both inference section headings exist (use heading role to avoid Notes match)
    expect(
      screen.getByRole("heading", { name: /server side inference/i })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /client side inference/i })
    ).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Zero / Boundaries: Initial state
// ---------------------------------------------------------------------------
describe("App initial state", () => {
  it("Predict button is disabled when no image selected", async () => {
    // Given: app rendered and settled, no image
    await renderApp();
    // Then: Predict is disabled
    expect(screen.getByRole("button", { name: /predict/i })).toBeDisabled();
  });

  it("no predictions displayed initially", async () => {
    // Given: app rendered and settled
    await renderApp();
    // Then: no probability values shown
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
  });

  it("fetches classes from API on mount", async () => {
    // Given/When: app renders
    await renderApp();
    // Then: fetch was called with /api/classes
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/classes")
    );
  });
});

// ---------------------------------------------------------------------------
// Interfaces: User interactions
// ---------------------------------------------------------------------------
describe("App user interactions", () => {
  it("URL input accepts text", async () => {
    // Given: app rendered and settled
    const user = userEvent.setup();
    await renderApp();
    const input = screen.getByPlaceholderText(/enter image url/i);
    // When: user types a URL
    await user.type(input, "https://example.com/food.jpg");
    // Then: input has the typed value
    expect(input).toHaveValue("https://example.com/food.jpg");
  });

  it("Clear button resets URL input", async () => {
    // Given: app with URL typed
    const user = userEvent.setup();
    await renderApp();
    const input = screen.getByPlaceholderText(/enter image url/i);
    await user.type(input, "https://example.com/food.jpg");
    // When: Clear is clicked
    await user.click(screen.getByRole("button", { name: /clear/i }));
    // Then: input is cleared
    expect(input).toHaveValue("");
  });

  it("Load Browser Model button shows Not loaded initially", async () => {
    // Given: app rendered and settled
    await renderApp();
    // Then: button text includes "Not loaded"
    expect(
      screen.getByRole("button", { name: /load browser model/i })
    ).toHaveTextContent(/not loaded/i);
  });
});
