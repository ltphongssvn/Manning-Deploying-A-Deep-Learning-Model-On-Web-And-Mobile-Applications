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
 * - One: single prediction flow
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import App from "./App";

// Mock TensorFlow.js — jsdom has no WebGL
const mockPredict = vi.fn();
const mockDispose = vi.fn();
const mockData = vi.fn();
const mockLoadGraphModel = vi.fn();

vi.mock("@tensorflow/tfjs", () => ({
  loadGraphModel: (...args: unknown[]) => mockLoadGraphModel(...args),
  tidy: (fn: () => unknown) => fn(),
  browser: {
    fromPixels: vi.fn(() => ({
      resizeBilinear: vi.fn().mockReturnThis(),
      toFloat: vi.fn(() => ({
        div: vi.fn(() => ({
          sub: vi.fn(() => ({
            expandDims: vi.fn(() => ({
              dispose: mockDispose,
            })),
          })),
        })),
      })),
    })),
  },
}));

// Mock fetch globally
const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

// Mock URL.createObjectURL for file preview
globalThis.URL.createObjectURL = vi.fn(() => "blob:http://test/fake-blob");
globalThis.URL.revokeObjectURL = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  mockFetch.mockResolvedValue({
    ok: true,
    json: async () => ({ classes: ["apple_pie", "caesar_salad", "falafel"] }),
  });
  mockLoadGraphModel.mockReset();
  mockPredict.mockReset();
  mockData.mockReset();
  mockDispose.mockReset();
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
    await renderApp();
    expect(
      screen.getByRole("heading", { name: /classify food image/i })
    ).toBeInTheDocument();
  });

  it("displays URL input field", async () => {
    await renderApp();
    expect(screen.getByPlaceholderText(/enter image url/i)).toBeInTheDocument();
  });

  it("displays file upload input", async () => {
    await renderApp();
    const fileInput = document.querySelector('input[type="file"]');
    expect(fileInput).toBeInTheDocument();
  });

  it("displays Predict button", async () => {
    await renderApp();
    expect(screen.getByRole("button", { name: /predict/i })).toBeInTheDocument();
  });

  it("displays Clear button", async () => {
    await renderApp();
    expect(screen.getByRole("button", { name: /clear/i })).toBeInTheDocument();
  });

  it("displays Load Browser Model button", async () => {
    await renderApp();
    expect(
      screen.getByRole("button", { name: /load browser model/i })
    ).toBeInTheDocument();
  });

  it("displays Server and Client inference headings", async () => {
    await renderApp();
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
    await renderApp();
    expect(screen.getByRole("button", { name: /predict/i })).toBeDisabled();
  });

  it("no predictions displayed initially", async () => {
    await renderApp();
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
  });

  it("fetches classes from API on mount", async () => {
    await renderApp();
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/classes")
    );
  });

  it("uses fallback classes when fetch fails", async () => {
    // Given: fetch rejects
    mockFetch.mockRejectedValueOnce(new Error("Network error"));
    render(<App />);
    // Then: app still renders (uses fallback classes)
    await waitFor(() => {
      expect(
        screen.getByRole("heading", { name: /classify food image/i })
      ).toBeInTheDocument();
    });
  });
});

// ---------------------------------------------------------------------------
// Interfaces: User interactions
// ---------------------------------------------------------------------------
describe("App user interactions", () => {
  it("URL input accepts text", async () => {
    const user = userEvent.setup();
    await renderApp();
    const input = screen.getByPlaceholderText(/enter image url/i);
    await user.type(input, "https://example.com/food.jpg");
    expect(input).toHaveValue("https://example.com/food.jpg");
  });

  it("Clear button resets URL input", async () => {
    const user = userEvent.setup();
    await renderApp();
    const input = screen.getByPlaceholderText(/enter image url/i);
    await user.type(input, "https://example.com/food.jpg");
    await user.click(screen.getByRole("button", { name: /clear/i }));
    expect(input).toHaveValue("");
  });

  it("Load Browser Model button shows Not loaded initially", async () => {
    await renderApp();
    expect(
      screen.getByRole("button", { name: /load browser model/i })
    ).toHaveTextContent(/not loaded/i);
  });

  it("file upload sets preview and enables Predict", async () => {
    const user = userEvent.setup();
    await renderApp();
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    const testFile = new File(["fake-image"], "test.jpg", { type: "image/jpeg" });
    await user.upload(fileInput, testFile);
    // Then: preview image appears and Predict is enabled
    await waitFor(() => {
      expect(screen.getByAltText("Preview")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /predict/i })).not.toBeDisabled();
    });
  });

  it("URL input enables Predict button", async () => {
    const user = userEvent.setup();
    await renderApp();
    const input = screen.getByPlaceholderText(/enter image url/i);
    await user.type(input, "https://example.com/food.jpg");
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /predict/i })).not.toBeDisabled();
    });
  });

  it("Clear button clears file upload and disables Predict", async () => {
    const user = userEvent.setup();
    await renderApp();
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    const testFile = new File(["fake-image"], "test.jpg", { type: "image/jpeg" });
    await user.upload(fileInput, testFile);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /predict/i })).not.toBeDisabled();
    });
    await user.click(screen.getByRole("button", { name: /clear/i }));
    expect(screen.getByRole("button", { name: /predict/i })).toBeDisabled();
  });
});

// ---------------------------------------------------------------------------
// Server prediction flow
// ---------------------------------------------------------------------------
describe("Server prediction", () => {
  it("sends file upload to /api/predict and displays results", async () => {
    const user = userEvent.setup();
    // Given: mock predict returns results
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ classes: ["apple_pie", "caesar_salad", "falafel"] }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          predictions: [
            { class: "falafel", probability: 0.9 },
            { class: "caesar_salad", probability: 0.08 },
            { class: "apple_pie", probability: 0.02 },
          ],
          inference_time_ms: 150.5,
        }),
      });
    await renderApp();
    // When: upload file and click Predict
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    const testFile = new File(["fake-image"], "test.jpg", { type: "image/jpeg" });
    await user.upload(fileInput, testFile);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /predict/i })).not.toBeDisabled();
    });
    await user.click(screen.getByRole("button", { name: /predict/i }));
    // Then: results displayed with class names
    await waitFor(() => {
      expect(screen.getByText("falafel")).toBeInTheDocument();
      expect(screen.getByText(/90\.00%/)).toBeInTheDocument();
    });
  });

  it("sends URL to /api/predict_url and displays results", async () => {
    const user = userEvent.setup();
    // Given: mock returns prediction results
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ classes: ["apple_pie", "caesar_salad", "falafel"] }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          predictions: [
            { class: "caesar_salad", probability: 0.85 },
            { class: "falafel", probability: 0.1 },
            { class: "apple_pie", probability: 0.05 },
          ],
          inference_time_ms: 200.0,
        }),
      });
    await renderApp();
    // When: enter URL and click Predict
    const input = screen.getByPlaceholderText(/enter image url/i);
    await user.type(input, "https://example.com/food.jpg");
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /predict/i })).not.toBeDisabled();
    });
    await user.click(screen.getByRole("button", { name: /predict/i }));
    // Then: results displayed with class names
    await waitFor(() => {
      expect(screen.getByText("caesar_salad")).toBeInTheDocument();
      expect(screen.getByText(/85\.00%/)).toBeInTheDocument();
    });
  });

  it("handles server prediction error gracefully", async () => {
    const user = userEvent.setup();
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    // Given: predict fetch rejects
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ classes: ["apple_pie", "caesar_salad", "falafel"] }),
      })
      .mockRejectedValueOnce(new Error("Server error"));
    await renderApp();
    // When: upload and predict
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    const testFile = new File(["fake-image"], "test.jpg", { type: "image/jpeg" });
    await user.upload(fileInput, testFile);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /predict/i })).not.toBeDisabled();
    });
    await user.click(screen.getByRole("button", { name: /predict/i }));
    // Then: error logged, no crash
    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalled();
    });
    consoleSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// Browser model loading
// ---------------------------------------------------------------------------
describe("Browser model loading", () => {
  it("loads TF.js model when Load Browser Model clicked", async () => {
    const user = userEvent.setup();
    const mockModel = { predict: mockPredict };
    mockLoadGraphModel.mockResolvedValueOnce(mockModel);
    await renderApp();
    // When: click Load Browser Model
    await user.click(screen.getByRole("button", { name: /load browser model/i }));
    // Then: model loaded, status shows Ready
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /load browser model/i })
      ).toHaveTextContent(/ready/i);
    });
  });

  it("shows Failed to load on model load error", async () => {
    const user = userEvent.setup();
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    mockLoadGraphModel.mockRejectedValueOnce(new Error("Load failed"));
    await renderApp();
    // When: click Load Browser Model (will fail)
    await user.click(screen.getByRole("button", { name: /load browser model/i }));
    // Then: status shows Failed
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /load browser model/i })
      ).toHaveTextContent(/failed to load/i);
    });
    consoleSpy.mockRestore();
  });

  it("disables Load Browser Model button after model loaded", async () => {
    const user = userEvent.setup();
    const mockModel = { predict: mockPredict };
    mockLoadGraphModel.mockResolvedValueOnce(mockModel);
    await renderApp();
    await user.click(screen.getByRole("button", { name: /load browser model/i }));
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /load browser model/i })
      ).toBeDisabled();
    });
  });
});

// ---------------------------------------------------------------------------
// ResultTable rendering
// ---------------------------------------------------------------------------
describe("ResultTable displays predictions", () => {
  it("shows class names and probabilities in table", async () => {
    const user = userEvent.setup();
    // Given: server returns predictions
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ classes: ["apple_pie", "caesar_salad", "falafel"] }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          predictions: [
            { class: "falafel", probability: 0.9 },
            { class: "caesar_salad", probability: 0.08 },
            { class: "apple_pie", probability: 0.02 },
          ],
          inference_time_ms: 100,
        }),
      });
    await renderApp();
    // When: upload and predict
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    const testFile = new File(["fake-image"], "test.jpg", { type: "image/jpeg" });
    await user.upload(fileInput, testFile);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /predict/i })).not.toBeDisabled();
    });
    await user.click(screen.getByRole("button", { name: /predict/i }));
    // Then: class names appear in results
    await waitFor(() => {
      expect(screen.getByText("falafel")).toBeInTheDocument();
      expect(screen.getByText("caesar_salad")).toBeInTheDocument();
      expect(screen.getByText("apple_pie")).toBeInTheDocument();
    });
  });

  it("shows inference time in results", async () => {
    const user = userEvent.setup();
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ classes: ["apple_pie", "caesar_salad", "falafel"] }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          predictions: [
            { class: "falafel", probability: 0.9 },
            { class: "caesar_salad", probability: 0.08 },
            { class: "apple_pie", probability: 0.02 },
          ],
          inference_time_ms: 250.75,
        }),
      });
    await renderApp();
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    const testFile = new File(["fake-image"], "test.jpg", { type: "image/jpeg" });
    await user.upload(fileInput, testFile);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /predict/i })).not.toBeDisabled();
    });
    await user.click(screen.getByRole("button", { name: /predict/i }));
    await waitFor(() => {
      expect(screen.getByText(/250\.75/)).toBeInTheDocument();
    });
  });
});
