import { trainMnistModel } from "./training/mnist-model.js";
import { fileURLToPath } from "url";
import express from "express";
import path from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;

let neuralNetworkInstance = null;

// Serve static files from public directory
app.use(express.static(path.join(__dirname, "../public")));

app.get("/train", (req, res) => {
  neuralNetworkInstance = trainMnistModel();
  return res.json({ metrics: neuralNetworkInstance.metrics.getMetrics() });
});

app.listen(port, () => {
  console.log(`Server is listening on port ${port}`);
});
