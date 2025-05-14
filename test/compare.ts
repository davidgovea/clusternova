import HDBSCAN, { VectorPoint } from '../src/hdbscan';
import * as process from 'process'; // Import process for hrtime

// To run: npx ts-node compare.ts
interface Point {
  id: string;
  vector: number[];
}

type DistanceMetric = "euclidean" | "manhattan" | "cosine"; // add other metrics as needed

interface ComparisonMetrics {
  tsDurationMs: string;
  tsClusterCount: number;
  tsOutlierCount: number;
  tsRssMb: string; // Added for TS RSS
  tsHeapTotalMb: string; // Added for TS HeapTotal
  tsHeapUsedMb: string; // Added for TS HeapUsed
  scikitDurationMs: string | null;
  scikitClusterCount: number | null;
  scikitOutlierCount: number | null;
  scikitRssMb: string | null; // Added for Scikit RSS
  agreementPercentage: number | null;
  error?: string;
}

async function compareWithScikit(
  points: Point[],
  minPoints: number,
  distanceMetric: DistanceMetric
): Promise<ComparisonMetrics> {

  switch (distanceMetric) {
    case "cosine":
      break;
    default:
      throw new Error(`Unsupported distance metric: ${distanceMetric}`);
  }
  // Run your implementation
  const tsStartTime = process.hrtime();
  const hdbscan = new HDBSCAN(points, minPoints);
  const tsResults = hdbscan.run();
  const tsEndTime = process.hrtime(tsStartTime);
  const tsDurationNs = tsEndTime[0] * 1e9 + tsEndTime[1];
  const tsDurationMs = (tsDurationNs / 1e6).toFixed(2);
  const tsClusterCount = tsResults.clusters.length;
  const tsOutlierCount = tsResults.outliers.length;

  // Capture TS memory usage
  const tsMemoryUsage = process.memoryUsage();
  const tsRssMb = (tsMemoryUsage.rss / 1024 / 1024).toFixed(2);
  const tsHeapTotalMb = (tsMemoryUsage.heapTotal / 1024 / 1024).toFixed(2);
  const tsHeapUsedMb = (tsMemoryUsage.heapUsed / 1024 / 1024).toFixed(2);

  // Run scikit-learn implementation
  try {
    // const pyStartTime = process.hrtime(); // Remove this line
    const response = await fetch("http://127.0.0.1:5000/cluster", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ points, minPoints, distanceMetric }),
    });
    // const pyEndTime = process.hrtime(pyStartTime); // Remove this line
    // const pyDurationNs = pyEndTime[0] * 1e9 + pyEndTime[1]; // Remove this line
    // const pyDurationMs = (pyDurationNs / 1e6).toFixed(2); // Remove this line


    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const scikitResults = await response.json();
    const pyDurationMs = scikitResults.duration_ms.toFixed(2); // Get duration from server response
    const scikitClusterCount = scikitResults.clusters.length;
    const scikitOutlierCount = scikitResults.outliers.length;

    // Enhanced parsing and logging for scikitRssMb
    let scikitRssMb: string | null = null;
    if (scikitResults.memory_rss_mb != null) { // Check if the key exists and is not null/undefined
        if (typeof scikitResults.memory_rss_mb === 'number' && isFinite(scikitResults.memory_rss_mb)) {
            scikitRssMb = scikitResults.memory_rss_mb.toFixed(2);
        } else {
            // Log an error if the field is present but not a valid finite number
            console.error(`[compare.ts] Error: Scikit-learn server returned 'memory_rss_mb', but it was not a finite number. Value received: ${JSON.stringify(scikitResults.memory_rss_mb)}`);
            // scikitRssMb remains null, which will lead to "N/A" in the CSV
        }
    } else {
        // Log a warning if the field is missing (null, undefined, or not present in the response object)
        console.warn(`[compare.ts] Warning: Scikit-learn server response did not include 'memory_rss_mb' or it was null/undefined. Check the Python server ('compare_server.py') for 'psutil' errors or ensure 'memory_rss_mb' is correctly added to the JSON response.`);
        // scikitRssMb remains null, which will lead to "N/A" in the CSV
    }

    // Compare results
    // console.log(`TypeScript Implementation (took ${tsDurationMs} ms):`);
    // console.log(
    //   "Clusters:",
    //   tsResults.clusters.map((cluster) => cluster.map((p) => p.id))
    // );
    // console.log(
    //   "Outliers:",
    //   tsResults.outliers.map((p) => p.id)
    // );

    // console.log(`\nScikit-learn Implementation (algorithmic time: ${pyDurationMs} ms):`);
    // console.log("Clusters:", scikitResults.clusters);
    // console.log("Outliers:", scikitResults.outliers);

    // Calculate agreement percentage
    const totalPoints = points.length;
    let agreements = 0;

    // Create maps of point assignments for both implementations
    const tsAssignments = new Map<string, number>();
    tsResults.clusters.forEach((cluster, i) => {
      cluster.forEach((point) => tsAssignments.set(point.id, i));
    });
    tsResults.outliers.forEach((point) => tsAssignments.set(point.id, -1));

    const scikitAssignments = new Map<string, number>();
    scikitResults.clusters.forEach((cluster: any, i: number) => {
      cluster.forEach((id: string) => scikitAssignments.set(id, i));
    });
    scikitResults.outliers.forEach((id: string) =>
      scikitAssignments.set(id, -1)
    );

    // Count points that are clustered similarly
    points.forEach((point) => {
      const tsCluster = tsAssignments.get(point.id);
      const scikitCluster = scikitAssignments.get(point.id);

      // Both marked as outliers or both in same cluster
      if (tsCluster === -1 && scikitCluster === -1) {
        agreements++;
      } else if (tsCluster !== -1 && scikitCluster !== -1) {
        // Check if points that are clustered together in one implementation
        // are also clustered together in the other
        const tsClusterPoints = tsResults.clusters[tsCluster!].map(
          (point) => point.id
        );
        const scikitClusterPoints = scikitResults.clusters[scikitCluster!];

        const inSameCluster = tsClusterPoints.every((id) =>
          scikitClusterPoints.includes(id)
        );

        if (inSameCluster) agreements++;
      }
    });

    const agreementPercentage = (agreements / totalPoints) * 100;
    // console.log(`\nAgreement percentage: ${agreementPercentage.toFixed(2)}%`);

    return {
      tsDurationMs,
      tsClusterCount,
      tsOutlierCount,
      tsRssMb, // Add TS RSS
      tsHeapTotalMb, // Add TS HeapTotal
      tsHeapUsedMb, // Add TS HeapUsed
      scikitDurationMs: pyDurationMs,
      scikitClusterCount,
      scikitOutlierCount,
      scikitRssMb, // Add Scikit RSS
      agreementPercentage,
    };
  } catch (error: any) {
    // console.error("Error comparing with scikit-learn:", error);
    return {
      tsDurationMs,
      tsClusterCount,
      tsOutlierCount,
      tsRssMb, // Add TS RSS even on Scikit error
      tsHeapTotalMb, // Add TS HeapTotal even on Scikit error
      tsHeapUsedMb, // Add TS HeapUsed even on Scikit error
      scikitDurationMs: null,
      scikitClusterCount: null,
      scikitOutlierCount: null,
      scikitRssMb: null, // Scikit RSS is null on error
      agreementPercentage: null,
      error: error.message || "Unknown error during Scikit comparison",
    };
  }
}

const minPoints = 10;

// Test cases
const testCases = [
  {
    name: "10 vectors",
    points: Array(10)
      .fill(0)
      .map((_, i) => ({
        id: i.toString(),
        vector: Array(1536)
          .fill(0)
          .map(() => Math.random() * 10),
      })),
    minPoints: 2,
  },
  {
    name: "50 vectors",
    points: Array(50)
      .fill(0)
      .map((_, i) => ({
        id: i.toString(),
        vector: Array(1536)
          .fill(0)
          .map(() => Math.random() * 10),
      })),
    minPoints: 5,
  },
  {
    name: "100 vectors",
    points: Array(100)
      .fill(0)
      .map((_, i) => ({
        id: i.toString(),
        vector: Array(1536)
          .fill(0)
          .map(() => Math.random() * 10),
      })),
    minPoints,
  },
  {
    name: "500 vectors",
    points: Array(500)
      .fill(0)
      .map((_, i) => ({
        id: i.toString(),
        vector: Array(1536)
          .fill(0)
          .map(() => Math.random() * 10),
      })),
    minPoints,
  },
  {
    name: "1000 vectors",
    points: Array(1000)
      .fill(0)
      .map((_, i) => ({
        id: i.toString(),
        vector: Array(1536)
          .fill(0)
          .map(() => Math.random() * 10),
      })),
    minPoints,
  },
  {
    name: "2000 vectors",
    points: Array(2000)
      .fill(0)
      .map((_, i) => ({
        id: i.toString(),
        vector: Array(1536)
          .fill(0)
          .map(() => Math.random() * 10),
      })),
    minPoints,
  },
  {
    name: "5000 vectors",
    points: Array(5000)
      .fill(0)
      .map((_, i) => ({
        id: i.toString(),
        vector: Array(1536)
          .fill(0)
          .map(() => Math.random() * 10),
      })),
    minPoints,
  },
  // {
  //   name: "10000 vectors",
  //   points: Array(10000)
  //     .fill(0)
  //     .map((_, i) => ({
  //       id: i.toString(),
  //       vector: Array(1536)
  //         .fill(0)
  //         .map(() => Math.random() * 10),
  //     })),
  //   minPoints,
  // },
  // {
  //   name: "20000 vectors",
  //   points: Array(20000)
  //     .fill(0)
  //     .map((_, i) => ({
  //       id: i.toString(),
  //       vector: Array(1536)
  //         .fill(0)
  //         .map(() => Math.random() * 10),
  //     })),
  //   minPoints,
  // },
];

async function runTests() {
  console.log("Test Name,TS Duration (ms),TS RSS (MB),TS HeapTotal (MB),TS HeapUsed (MB),Scikit Duration (ms),Scikit RSS (MB),TS Clusters,TS Outliers,Scikit Clusters,Scikit Outliers,Agreement (%)");
  for (const testCase of testCases) {
    // console.log(`\nRunning test: ${testCase.name}`); // Optional: keep for verbose logging or log to stderr
    const metrics = await compareWithScikit(testCase.points, testCase.minPoints, "cosine");

    const scikitDuration = metrics.scikitDurationMs ?? "N/A";
    const scikitClusters = metrics.scikitClusterCount ?? "N/A";
    const scikitOutliers = metrics.scikitOutlierCount ?? "N/A";
    const scikitRss = metrics.scikitRssMb ?? "N/A"; // Get Scikit RSS
    const agreement = metrics.agreementPercentage !== null ? metrics.agreementPercentage.toFixed(2) : "N/A";

    let csvLine = `${testCase.name},${metrics.tsDurationMs},${metrics.tsRssMb},${metrics.tsHeapTotalMb},${metrics.tsHeapUsedMb},${scikitDuration},${scikitRss},${metrics.tsClusterCount},${metrics.tsOutlierCount},${scikitClusters},${scikitOutliers},${agreement}`;
    if (metrics.error) {
      csvLine += `,Error: ${metrics.error}`;
    }
    console.log(csvLine);
  }
}

runTests();