import HDBSCAN, { VectorPoint } from '../src/hdbscan';
import * as process from 'process'; // For process.hrtime, process.memoryUsage, process.argv, process.exit
import * as fs from 'fs';

// Helper function to generate random high-dimensional data
function generateData(numPoints: number, numDimensions: number): VectorPoint[] {
    console.log(`Generating ${numPoints} points with ${numDimensions} dimensions...`);
    const data: VectorPoint[] = [];
    const startTime = process.hrtime();
    for (let i = 0; i < numPoints; i++) {
        const vector: number[] = new Array(numDimensions);
        for (let j = 0; j < numDimensions; j++) {
            vector[j] = Math.random(); // Simple random data [0, 1)
        }
        data.push({ id: `point_${i}`, vector });
    }
    const endTime = process.hrtime(startTime);
    const durationNs = endTime[0] * 1e9 + endTime[1];
    console.log(`Data generation took ${(durationNs / 1e6).toFixed(2)} ms.`);
    return data;
}

// Function to run and benchmark HDBSCAN
async function runBenchmark(points: VectorPoint[], mpts: number, scenarioName: string) {
    console.log(`\n--- Starting Benchmark Scenario: ${scenarioName} ---`);
    console.log(`Parameters: ${points.length} points, ${points[0]?.vector.length || 0} dimensions, mpts=${mpts}`);

    if (points.length === 0) {
        console.log("No points to process for this scenario.");
        return;
    }
    if (mpts <= 0) {
        console.error(`Invalid mpts value (${mpts}) for scenario "${scenarioName}". mpts must be positive. Skipping.`);
        return;
    }
    if (points.length <= mpts && points.length > 0) { // mpts must be < N for core distance calculation
        console.warn(`Warning for scenario "${scenarioName}": mpts (${mpts}) is not less than the number of data points (${points.length}). This might lead to errors or unexpected behavior in HDBSCAN. Adjusting mpts to ${Math.max(1, points.length -1)} or skipping if not possible.`);
        // Depending on strictness, either skip or adjust. HDBSCAN itself throws error if mpts >= N.
        // For benchmark, it's better to highlight this.
        if (points.length === 1 && mpts >=1) { // HDBSCAN requires N > mpts
             console.error(`Scenario "${scenarioName}" cannot run: mpts (${mpts}) must be less than N (${points.length}). Skipping.`);
             return;
        }
         // Let HDBSCAN's internal checks handle it, or adjust mpts:
        // mpts = Math.max(1, points.length - 1);
        // console.log(`Adjusted mpts to ${mpts} for scenario "${scenarioName}".`);
    }


    const hdbscan = new HDBSCAN(points, mpts);

    const benchmarkStartTime = process.hrtime();
    try {
        const results = hdbscan.run();
        const benchmarkEndTime = process.hrtime(benchmarkStartTime);
        const durationNs = benchmarkEndTime[0] * 1e9 + benchmarkEndTime[1];
        const durationMs = durationNs / 1e6;

        console.log(`HDBSCAN for "${scenarioName}" completed in ${durationMs.toFixed(2)} ms`);
        console.log(`Found ${results.clusters.length} clusters and ${results.outliers.length} outliers.`);

        const memoryUsage = process.memoryUsage();
        console.log(`Memory usage: RSS=${(memoryUsage.rss / 1024 / 1024).toFixed(2)}MB, HeapTotal=${(memoryUsage.heapTotal / 1024 / 1024).toFixed(2)}MB, HeapUsed=${(memoryUsage.heapUsed / 1024 / 1024).toFixed(2)}MB`);

    } catch (error) {
        console.error(`Benchmark scenario "${scenarioName}" failed:`, error);
        const benchmarkEndTime = process.hrtime(benchmarkStartTime);
        const durationNs = benchmarkEndTime[0] * 1e9 + benchmarkEndTime[1];
        const durationMs = durationNs / 1e6;
        console.log(`Failed after ${durationMs.toFixed(2)} ms`);
    }
}

// Main benchmark execution
async function main() {
    console.log("HDBSCAN Performance Benchmark Suite");
    console.log("===================================");
    console.log("NOTE: You might need to install 'apache-arrow': npm install apache-arrow OR yarn add apache-arrow");

    const scenarios = [
        { name: "Tiny test (check overhead)", points: 10, dimensions: 10, mpts: 2, type: "generate" },
        { name: "Small test", points: 100, dimensions: 128, mpts: 5, type: "generate" },
        { name: "User workload dimensions (small N)", points: 500, dimensions: 1536, mpts: 5, type: "generate" },
        { name: "User workload dimensions (medium N)", points: 1000, dimensions: 1536, mpts: 10, type: "generate" },
        { name: "User workload dimensions (larger N)", points: 2000, dimensions: 1536, mpts: 10, type: "generate" },
    ];

    for (const scenario of scenarios) {
        let data: VectorPoint[];
        // @ts-ignore
        if (scenario.type === "parquet") {
            // @ts-ignore
            data = scenario.points as VectorPoint[]; // Data is already loaded
        // @ts-ignore
        } else if (scenario.type === "generate") {
            // @ts-ignore
            data = generateData(scenario.points, scenario.dimensions);
        } else {
            console.warn(`Unknown scenario type for ${scenario.name}. Skipping.`);
            continue;
        }
        
        // @ts-ignore
        await runBenchmark(data, scenario.mpts, scenario.name);

        const runLargeScenarioFlag = process.argv.includes('--run-large');
        if (runLargeScenarioFlag) {
            console.log("\nPreparing for large scenario (10k points, generated data)...");
            const largeGeneratedScenario = { name: "User workload (10k points, generated)", points: 10000, dimensions: 1536, mpts: 10 };
            const data = generateData(largeGeneratedScenario.points, largeGeneratedScenario.dimensions);
            await runBenchmark(data, largeGeneratedScenario.mpts, largeGeneratedScenario.name);
        } else {
            console.log("\nTo run the 10k point benchmark scenario (with generated data), pass the --run-large flag.");
        }

    console.log("\nTo run the 10k point benchmark scenario, pass the --run-large flag.");
    console.log("Example: npx ts-node test/benchmark.ts --run-large");
    console.log("For potentially better memory readings with GC: npx ts-node --expose-gc test/benchmark.ts --run-large");
}

console.log("\n===================================");
console.log("Benchmark suite finished.");
}

main().catch(error => {
    console.error("Unhandled error in main benchmark execution:", error);
    process.exit(1);
});

// How to run:
// 1. Standard run (most scenarios):
//    npx ts-node test/benchmark.ts
// 2. Run including the large 10k point scenario:
//    npx ts-node test/benchmark.ts --run-large
// 3. Run with manual garbage collection enabled (recommended for memory analysis, especially with --run-large):
//    npx ts-node --expose-gc test/benchmark.ts
//    npx ts-node --expose-gc test/benchmark.ts --run-large
