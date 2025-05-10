import * as tf from '@tensorflow/tfjs-node';
import { UnionFind, cosine } from './utils.ts';
export type DistanceFunction = (pointA: number[], pointB: number[]) => number;
export interface VectorPoint {
  id: string;
  vector: number[];
}

class HDBSCAN<T extends VectorPoint> {
  private X: T[];
  private mpts: number;
  private coreDistances: number[];
  private mstEdges: { from: string; to: string; weight: number }[];
  private idToObject: Map<string, T>;
  private idToIndex: Map<string, number>;
  private indexToId: string[];
  private distancesArray: number[][];
  private mrdArray: number[][];

  constructor(X: T[], mpts: number) {
    this.X = X;
    this.mpts = mpts;
    this.coreDistances = [];
    this.mstEdges = [];
    this.idToObject = new Map();
    this.X.forEach((p) => {
      this.idToObject.set(p.id, p);
    });
    const N = this.X.length;
    this.indexToId = this.X.map(p => p.id);
    this.idToIndex = new Map(this.indexToId.map((id, index) => [id, index]));
    this.distancesArray = [];
    this.mrdArray = [];
  }

  run(): { clusters: T[][]; outliers: T[] } {
    if (this.X.length === 0) {
      return { clusters: [], outliers: [] };
    }
    const N = this.X.length;

    if (this.mpts <= 0) {
        throw new Error("mpts must be positive.");
    }
    if (this.mpts >= N && N > 0) {
        throw new Error(`mpts (${this.mpts}) must be less than the number of data points (${N}) for core distance calculation.`);
    }

    try {
      console.time('HDBSCAN: Distance Matrix');
      const vectorsData = this.X.map(p => p.vector);
      const vectorsTensor = tf.tensor2d(vectorsData);
      const normsTensor = tf.norm(vectorsTensor, 'euclidean', 1, true);
      const zeroNormMask = tf.equal(normsTensor, tf.scalar(0));
      const safeNorms = tf.where(zeroNormMask, tf.onesLike(normsTensor), normsTensor);
      const normalizedVectors = vectorsTensor.div(safeNorms);
      const dotProducts = tf.matMul(normalizedVectors, normalizedVectors, false, true);
      const distancesTensor = tf.sub(tf.scalar(1), dotProducts);
      this.distancesArray = distancesTensor.arraySync();

      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          if (isNaN(this.distancesArray[i][j])) {
            this.distancesArray[i][j] = 1.0;
          }
        }
        this.distancesArray[i][i] = 0.0;
      }
      console.timeEnd('HDBSCAN: Distance Matrix');

      console.time('HDBSCAN: Core Distances');
      const negDistancesTensor = distancesTensor.neg();
      const { values: smallestNegatedDistances } = tf.topk(negDistancesTensor, this.mpts + 1, true);
      const coreDistancesTensor = smallestNegatedDistances.gather([this.mpts], 1).neg();
      this.coreDistances = Array.from(coreDistancesTensor.dataSync());
      console.timeEnd('HDBSCAN: Core Distances');

      console.time('HDBSCAN: MRD Matrix');
      const coreDistancesTensor2D = tf.tensor2d(this.coreDistances, [N, 1]);
      const coreDistancesTiled = coreDistancesTensor2D.tile([1, N]);
      const coreDistancesTransposed = coreDistancesTensor2D.tile([1, N]).transpose();
      const maxCoreDistances = tf.maximum(coreDistancesTiled, coreDistancesTransposed);
      const mrdTensor = tf.maximum(maxCoreDistances, distancesTensor);
      this.mrdArray = mrdTensor.arraySync();
      console.timeEnd('HDBSCAN: MRD Matrix');

      vectorsTensor.dispose();
      normsTensor.dispose();
      zeroNormMask.dispose();
      safeNorms.dispose();
      normalizedVectors.dispose();
      dotProducts.dispose();
      distancesTensor.dispose();
      negDistancesTensor.dispose();
      smallestNegatedDistances.dispose();
      coreDistancesTensor.dispose();
      coreDistancesTensor2D.dispose();
      coreDistancesTiled.dispose();
      coreDistancesTransposed.dispose();
      maxCoreDistances.dispose();
      mrdTensor.dispose();

      console.time('HDBSCAN: MST Computation');
      this._computeMST();
      console.timeEnd('HDBSCAN: MST Computation');

      console.time('HDBSCAN: Hierarchy Extraction');
      const { clusters, outliers } = this._extractHDBSCANHierarchy();
      console.timeEnd('HDBSCAN: Hierarchy Extraction');

      const clustersWithOriginalObjs = clusters.map((cluster) =>
        cluster.map((id) => this.idToObject.get(id)!)
      );
      const outliersWithOriginalObjs = outliers.map(
        (id) => this.idToObject.get(id)!
      );

      return {
        clusters: clustersWithOriginalObjs,
        outliers: outliersWithOriginalObjs,
      };
    } catch (e) {
      console.error("Error in HDBSCAN:", e);
      throw e;
    }
  }

  private getDistance(i: number, j: number): number {
    if (i < 0 || i >= this.X.length || j < 0 || j >= this.X.length) {
        throw new Error("Invalid indices for getDistance");
    }
    return this.distancesArray[i][j];
  }

  private getMRD(i: number, j: number): number {
    if (i < 0 || i >= this.X.length || j < 0 || j >= this.X.length) {
      throw new Error("Invalid indices for getMRD");
    }
    return this.mrdArray[i][j];
  }

  private _computeMST() {
    const N = this.X.length;
    if (N === 0) return;

    const key = new Array(N).fill(Infinity);
    const parent = new Array(N).fill(-1);
    const inMST = new Array(N).fill(false);
    this.mstEdges = [];

    key[0] = 0; // Start with the first vertex

    for (let count = 0; count < N; count++) {
      let minKey = Infinity;
      let u = -1;

      // Find vertex with minimum key not yet in MST
      for (let i = 0; i < N; i++) {
        if (!inMST[i] && key[i] < minKey) {
          minKey = key[i];
          u = i;
        }
      }

      if (u === -1) {
        // Should not happen in a connected graph (MRD matrix implies complete graph)
        console.warn("MST construction failed to find next vertex. Graph might be disconnected.");
        break;
      }

      inMST[u] = true;

      // Add edge to MST, except for the first vertex (which has no parent)
      if (parent[u] !== -1) {
        this.mstEdges.push({
          from: this.indexToId[parent[u]],
          to: this.indexToId[u],
          weight: key[u],
        });
      }

      // Update keys of adjacent vertices
      for (let v = 0; v < N; v++) {
        if (!inMST[v] && this.mrdArray[u][v] < key[v]) {
          parent[v] = u;
          key[v] = this.mrdArray[u][v];
        }
      }
    }
    this.mstEdges.sort((a, b) => a.weight - b.weight);
  }

  private _extractHDBSCANHierarchy() {
    const uf = new UnionFind(this.X.map((point) => point.id));
    const hierarchy: {
      childrenClusters: number[] | null;
      elements: string[];
      lambdaPs: number[];
      lambdaMin: number | null;
      lambdaMax: number;
    }[] = [];
    const pointToHierarchyIndex = new Map();
    const currentGroups = new Map();
    for (const key of this.X.map((point) => point.id)) {
      currentGroups.set(key, [key]);
    }

    this.mstEdges.forEach((edge) => {
      const { from, to, weight } = edge;
      const rootFrom = uf.find(from);
      const rootTo = uf.find(to);
      const sizeFrom = currentGroups.get(rootFrom).length;
      const sizeTo = currentGroups.get(rootTo).length;
      const newSize = sizeFrom + sizeTo;

      uf.union(from, to);
      const newRoot = uf.find(from);
      const newElements = currentGroups
        .get(rootFrom)
        .concat(currentGroups.get(rootTo));

      if (newSize >= this.mpts && sizeFrom < this.mpts && sizeTo < this.mpts) {
        hierarchy.push({
          childrenClusters: null,
          elements: newElements,
          lambdaPs: new Array(newElements.length).fill(1 / weight),
          lambdaMin: null,
          lambdaMax: 1 / weight,
        });
        pointToHierarchyIndex.set(newRoot, hierarchy.length - 1);
      } else if (
        newSize >= this.mpts &&
        sizeFrom >= this.mpts &&
        sizeTo >= this.mpts
      ) {
        hierarchy.push({
          childrenClusters: [
            pointToHierarchyIndex.get(rootFrom),
            pointToHierarchyIndex.get(rootTo),
          ],
          elements: newElements,
          lambdaPs: new Array(newElements.length).fill(1 / weight),
          lambdaMin: null,
          lambdaMax: 1 / weight,
        });
        hierarchy[pointToHierarchyIndex.get(rootFrom)].lambdaMin = 1 / weight;
        hierarchy[pointToHierarchyIndex.get(rootTo)].lambdaMin = 1 / weight;
        pointToHierarchyIndex.set(newRoot, hierarchy.length - 1);
      } else if (newSize >= this.mpts) {
        if (pointToHierarchyIndex.get(newRoot) === undefined) {
          const existingIndex =
            pointToHierarchyIndex.get(rootFrom) ??
            pointToHierarchyIndex.get(rootTo);
          pointToHierarchyIndex.set(newRoot, existingIndex);
        }
        const updateCluster = hierarchy[pointToHierarchyIndex.get(newRoot)];
        updateCluster.elements = newElements;
        const mergeSize = sizeFrom < this.mpts ? sizeFrom : sizeTo;
        for (let i = 0; i < mergeSize; i++) {
          updateCluster.lambdaPs.push(1 / weight);
        }
      }
      currentGroups.set(newRoot, newElements);
      currentGroups.delete(newRoot === rootFrom ? rootTo : rootFrom);
    });

    hierarchy.pop();

    const stabilities = new Array(hierarchy.length).fill(0);
    for (let i = 0; i < hierarchy.length; i++) {
      for (let j = 0; j < hierarchy[i].lambdaPs.length; j++) {
        stabilities[i] +=
          hierarchy[i].lambdaPs[j] - (hierarchy[i].lambdaMin ?? 0);
      }
    }

    const isSelected = new Array(hierarchy.length).fill(false);
    const s_hat = new Array(hierarchy.length).fill(0);
    for (let i = 0; i < hierarchy.length; i++) {
      if (hierarchy[i].childrenClusters === null) {
        s_hat[i] = stabilities[i];
        isSelected[i] = true;
      } else {
        const i_left_child_index = hierarchy[i].childrenClusters![0];
        const i_right_child_index = hierarchy[i].childrenClusters![1];
        const i_left_child = s_hat[i_left_child_index];
        const i_right_child = s_hat[i_right_child_index];

        if (stabilities[i] < i_left_child + i_right_child) {
          s_hat[i] = i_left_child + i_right_child;
          isSelected[i] = false;
        } else {
          s_hat[i] = stabilities[i];
          isSelected[i] = true;
          isSelected[i_left_child_index] = false;
          isSelected[i_right_child_index] = false;
        }
      }
    }

    const clusters: Array<Array<string>> = [];
    const outliers: Array<string> = [];

    for (let i = 0; i < hierarchy.length; i++) {
      if (isSelected[i]) {
        clusters.push(hierarchy[i].elements);
      }
    }

    const allIds = new Set(this.X.map((point) => point.id));
    clusters.forEach((cluster) => {
      cluster.forEach((id) => {
        allIds.delete(id);
      });
    });
    outliers.push(...allIds);

    return { clusters, outliers };
  }
}

export function findCentralElements<T extends VectorPoint>(
  cluster: T[],
  n: number,
  distanceFunction: DistanceFunction = cosine
): (T & { distance: number })[] {
  if (n < 1) {
    throw new Error("Number of central elements must be at least 1");
  }
  if (cluster.length === 0) {
    throw new Error("Cannot find central elements of empty cluster");
  }

  const vectors = cluster.map((p) => p.vector);
  const dimensions = vectors[0].length;
  const centroid: number[] = new Array(dimensions).fill(0);

  for (const vector of vectors) {
    if (vector.length !== dimensions) {
      throw new Error("All vectors must have the same dimensions");
    }
    for (let i = 0; i < dimensions; i++) {
      centroid[i] += vector[i];
    }
  }

  for (let i = 0; i < dimensions; i++) {
    centroid[i] /= vectors.length;
  }

  return cluster
    .map((point) => ({
      ...point,
      distance: distanceFunction(point.vector, centroid),
    }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, n);
}

export default HDBSCAN;
