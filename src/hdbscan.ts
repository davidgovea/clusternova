export type DistanceFunction = (pointA: number[], pointB: number[]) => number;
export interface VectorPoint {
  id: string;
  vector: number[];
}
/**
 * HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) implementation
 * @template T - Type of data points, must include 'id' and 'vector' properties
 */
class HDBSCAN<T extends VectorPoint> {
  private X: T[];
  private mpts: number; // The minimum points to define a core point
  private coreDistances: number[]; // Array of core distances by index
  private mstEdges: { from: string; to: string; weight: number }[]; // mutual reachability graph as an adjacency list
  private distanceFunction: DistanceFunction;
  private idToObject: Map<string, T>;
  private idToIndex: Map<string, number>;
  private indexToId: string[];
  private typedVectors: Float64Array[];
  private norms: number[];
  private cumSum: Uint32Array;
  private distances: Float64Array;
  private mrgDistances: Float64Array;

  /**
   * Creates a new HDBSCAN instance
   * @param X - Array of data points to cluster
   * @param mpts - Minimum points required to form a dense region (minimum cluster size)
   * @param distanceFunction - Optional function to calculate distance between points (defaults to cosine distance)
   */
  constructor(X: T[], mpts: number, distanceFunction?: DistanceFunction) {
    this.X = X;
    this.mpts = mpts;
    this.coreDistances = [];
    this.mstEdges = [];
    this.distanceFunction = distanceFunction ?? cosine;
    this.mstEdges = [];
    this.distanceFunction = distanceFunction ?? cosine;
    this.idToObject = new Map();
    this.X.forEach((p) => {
      this.idToObject.set(p.id, p);
    });

    const N = this.X.length;
    this.indexToId = this.X.map(p => p.id);
    this.idToIndex = new Map(this.indexToId.map((id, index) => [id, index]));
    this.typedVectors = this.X.map(p => new Float64Array(p.vector));
    this.norms = this.typedVectors.map(vec => {
      let sum = 0.0;
      for (let i = 0; i < vec.length; i++) {
        sum += vec[i] * vec[i];
      }
      return Math.sqrt(sum);
    });
    this.cumSum = new Uint32Array(N);
    let sum = 0;
    for (let i = 0; i < N; i++) {
      this.cumSum[i] = sum;
      sum += N - i - 1;
    }
    this.distances = new Float64Array((N * (N - 1)) / 2);
    this.mrgDistances = new Float64Array((N * (N - 1)) / 2);
  }

  /**
   * Runs the HDBSCAN clustering algorithm
   * @returns Object containing clusters and outliers
   * @returns {T[][]} clusters - Array of clusters, where each cluster is an array of data points
   * @returns {T[]} outliers - Array of data points that don't belong to any cluster
   * @throws {Error} If an error occurs during clustering
   */
  run(): { clusters: T[][]; outliers: T[] } {
    if (this.X.length === 0) {
      return { clusters: [], outliers: [] };
    }
    try {
      this._computeAllNearestNeighbors();
      this._computeCoreDistances();
      this._constructMRG();
      this._computeMST();
      const { clusters, outliers } = this._extractHDBSCANHierarchy();

      // Transform IDs into original objects
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
      //rethrow
      throw e;
    }
  }

  private _computeAllNearestNeighbors() {
    const N = this.X.length;
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dist = this.computeDistance(i, j);
        const index = this.cumSum[i] + (j - i - 1);
        this.distances[index] = dist;
      }
    }
  }

  private computeDistance(i: number, j: number): number {
    const vecA = this.typedVectors[i];
    const vecB = this.typedVectors[j];
    const normA = this.norms[i];
    const normB = this.norms[j];
  
    if (normA === 0 || normB === 0) {
      return 1.0; // Cosine distance is 1 if one vector is zero (maximal dissimilarity)
    }
  
    let dotProduct = 0.0;
    const D = vecA.length; // Dimension of vectors
  
    // Unroll loop by 8 for performance
    const D_floor_8 = D - (D % 8);
    for (let k = 0; k < D_floor_8; k += 8) {
      dotProduct += vecA[k] * vecB[k] +
                    vecA[k+1] * vecB[k+1] +
                    vecA[k+2] * vecB[k+2] +
                    vecA[k+3] * vecB[k+3] +
                    vecA[k+4] * vecB[k+4] +
                    vecA[k+5] * vecB[k+5] +
                    vecA[k+6] * vecB[k+6] +
                    vecA[k+7] * vecB[k+7];
    }
  
    // Handle remaining elements if D is not a multiple of 8
    for (let k = D_floor_8; k < D; k++) {
      dotProduct += vecA[k] * vecB[k];
    }
  
    const similarity = dotProduct / (normA * normB);
    
    // Clamp similarity to [-1, 1] to handle potential floating point inaccuracies
    const clampedSimilarity = Math.max(-1.0, Math.min(1.0, similarity));
  
    return 1.0 - clampedSimilarity; // Cosine distance
  }

  private _computeCoreDistances() {
    const N = this.X.length;
    this.coreDistances = new Array(N);
    for (let i = 0; i < N; i++) {
      const distancesForI = [];
      for (let j = 0; j < i; j++) {
        const index = this.cumSum[j] + (i - j - 1);
        distancesForI.push(this.distances[index]);
      }
      for (let j = i + 1; j < N; j++) {
        const index = this.cumSum[i] + (j - i - 1);
        distancesForI.push(this.distances[index]);
      }
      distancesForI.sort((a, b) => a - b);
      if (distancesForI.length < this.mpts) {
        throw new Error("mpts is greater than the number of points in the dataset");
      }
      this.coreDistances[i] = distancesForI[this.mpts - 1];
    }
  }

  private _constructMRG() {
    const N = this.X.length;
    for (let i = 0; i < N; i++) {
      const coreDistI = this.coreDistances[i];
      for (let j = i + 1; j < N; j++) {
        const coreDistJ = this.coreDistances[j];
        const distIJ = this.getDistance(i, j);
        const mrd = Math.max(coreDistI, coreDistJ, distIJ);
        const index = this.cumSum[i] + (j - i - 1);
        this.mrgDistances[index] = mrd;
      }
    }
  }

  private getDistance(i: number, j: number): number {
    if (i > j) [i, j] = [j, i];
    const index = this.cumSum[i] + (j - i - 1);
    return this.distances[index];
  }

  private _computeMST() {
    const N = this.X.length;
    if (N === 0) return;
    const startIndex = 0;
    const pq = new PriorityQueue<{ from: number; to: number; weight: number }>((a, b) => a.weight < b.weight);
    for (let to = 0; to < N; to++) {
      if (to !== startIndex) {
        const weight = this.getMRD(startIndex, to);
        pq.enqueue({ from: startIndex, to, weight });
      }
    }
    const inMST = new Set([startIndex]);
    this.mstEdges = [];
    while (!pq.isEmpty()) {
      const edge = pq.dequeue();
      if (edge === null) continue;
      const { from, to, weight } = edge;
      if (!inMST.has(to)) {
        inMST.add(to);
        this.mstEdges.push({ from: this.indexToId[from], to: this.indexToId[to], weight });
        for (let next = 0; next < N; next++) {
          if (!inMST.has(next)) {
            const nextWeight = this.getMRD(to, next);
            pq.enqueue({ from: to, to: next, weight: nextWeight });
          }
        }
      }
    }
    this.mstEdges.sort((a, b) => a.weight - b.weight);
  }

  private getMRD(i: number, j: number): number {
    if (i > j) [i, j] = [j, i];
    const index = this.cumSum[i] + (j - i - 1);
    return this.mrgDistances[index];
  }

  private _extractHDBSCANHierarchy() {
    // Initialize the union-find structure
    const uf = new UnionFind(this.X.map((point) => point.id));

    // Array to store the hierarchy steps, where each element is an object:
    const hierarchy: {
      childrenClusters: number[] | null; // [two children index in the hierarchy (number)]
      elements: string[]; // [ids] (we also get the size of the cluster here),
      lambdaPs: number[];
      lambdaMin: number | null;
      lambdaMax: number;
    }[] = [];

    // map from point ID to its current highest index in hierarchy:
    const pointToHierarchyIndex = new Map();

    // to start, each point is in its own group (this is effectively our version of the MSText step)
    const currentGroups = new Map(); // Map of current groups (both clusters and noise points). key: id of root of the group, value: array of ids in the group
    for (const key of this.X.map((point) => point.id)) {
      currentGroups.set(key, [key]);
    }

    // Merge clusters based on sorted edges
    this.mstEdges.forEach((edge) => {
      const { from, to, weight } = edge;
      const rootFrom = uf.find(from);
      const rootTo = uf.find(to);
      const sizeFrom = currentGroups.get(rootFrom).length;
      const sizeTo = currentGroups.get(rootTo).length;
      const newSize = sizeFrom + sizeTo;

      // merge two noise points to form a new cluster!
      uf.union(from, to);
      //find what the root of the new cluster is
      const newRoot = uf.find(from);
      // console.log("newRoot", newRoot);
      const newElements = currentGroups
        .get(rootFrom)
        .concat(currentGroups.get(rootTo));

      if (newSize >= this.mpts && sizeFrom < this.mpts && sizeTo < this.mpts) {
        // merge two noise points to form a new cluster!

        // push the new cluster to the hierarchy
        hierarchy.push({
          childrenClusters: null, // first level cluster so no children
          elements: newElements,
          lambdaPs: new Array(newElements.length).fill(1 / weight),
          lambdaMin: null, // we don't know yet!
          lambdaMax: 1 / weight,
        });

        // update the root of the new cluster in the pointToHierarchyIndex map
        pointToHierarchyIndex.set(newRoot, hierarchy.length - 1);
      } else if (
        newSize >= this.mpts &&
        sizeFrom >= this.mpts &&
        sizeTo >= this.mpts
      ) {
        // merge two clusters to form a new cluster!

        // push the new cluster to the hierarchy
        hierarchy.push({
          childrenClusters: [
            pointToHierarchyIndex.get(rootFrom),
            pointToHierarchyIndex.get(rootTo),
          ],
          elements: newElements,
          lambdaPs: new Array(newElements.length).fill(1 / weight),
          lambdaMin: null, // we don't know yet!
          lambdaMax: 1 / weight,
        });

        //update the lambdaMin of the two children clusters
        hierarchy[pointToHierarchyIndex.get(rootFrom)].lambdaMin = 1 / weight;
        hierarchy[pointToHierarchyIndex.get(rootTo)].lambdaMin = 1 / weight;

        // update the root of the new cluster in the pointToHierarchyIndex map
        pointToHierarchyIndex.set(newRoot, hierarchy.length - 1);
      } else if (newSize >= this.mpts) {
        // merge a noise group with a cluster so a cluster grows bigger

        if (pointToHierarchyIndex.get(newRoot) === undefined) {
          // this means union find for some reason made the noise group the new root, so we just assign the other group's hierarchy index
          const existingIndex =
            pointToHierarchyIndex.get(rootFrom) ??
            pointToHierarchyIndex.get(rootTo);
          pointToHierarchyIndex.set(newRoot, existingIndex);
        }
        // find the existing cluster and modify it:
        const updateCluster = hierarchy[pointToHierarchyIndex.get(newRoot)];
        updateCluster.elements = newElements;
        const mergeSize = sizeFrom < this.mpts ? sizeFrom : sizeTo; // the size of the group that was noise
        for (let i = 0; i < mergeSize; i++) {
          updateCluster.lambdaPs.push(1 / weight);
        }
      }
      currentGroups.set(newRoot, newElements);
      currentGroups.delete(newRoot === rootFrom ? rootTo : rootFrom); // merged into newRoot, so the merged in root no longer considered
    });

    // remove last element of hierarchy array bc we don't care about the root
    hierarchy.pop();

    // creating and condensing the hierarchy tree is done! Now we calculate stabilities of every cluster:
    const stabilities = new Array(hierarchy.length).fill(0); // index is the index in the hierarchy array, value is the stability
    for (let i = 0; i < hierarchy.length; i++) {
      for (let j = 0; j < hierarchy[i].lambdaPs.length; j++) {
        stabilities[i] +=
          hierarchy[i].lambdaPs[j] - (hierarchy[i].lambdaMin ?? 0);
      }
    }

    // now we loop through the clusters again reverse topological order and set s_hat s.t.:
    //  s_hat(cluster_i) =
    //    {stabilities(cluster_i) iff cluster_i is leaf node
    //    {max(cluster_i, s_hat(cluster_i_left_child) + s_hat(cluster_i_right_child)) otherwise
    // and we select the cluster where its stability is greater than the sum of the s_hat values of its children

    const isSelected = new Array(hierarchy.length).fill(false); //index is the index in the hierarchy array, boolean value is whether we select it as a cluster
    const s_hat = new Array(hierarchy.length).fill(0); // index is the index in the hierarchy array, value is the s_hat value
    for (let i = 0; i < hierarchy.length; i++) {
      if (hierarchy[i].childrenClusters === null) {
        //cluster_i is leaf node
        s_hat[i] = stabilities[i];
        isSelected[i] = true;
      } else {
        const i_left_child_index = hierarchy[i].childrenClusters![0]; // since we remove the root, this must be defined
        const i_right_child_index = hierarchy[i].childrenClusters![1];
        const i_left_child = s_hat[i_left_child_index];
        const i_right_child = s_hat[i_right_child_index];

        if (stabilities[i] < i_left_child + i_right_child) {
          s_hat[i] = i_left_child + i_right_child;
          isSelected[i] = false;
        } else {
          s_hat[i] = stabilities[i];
          isSelected[i] = true;

          // unselect children here now that we've selected their parents
          isSelected[i_left_child_index] = false;
          isSelected[i_right_child_index] = false;
        }
      }
    }

    const clusters: Array<Array<string>> = []; // Each cluster is an array of string ids (string[][])
    const outliers: Array<string> = []; // List of outlier ids (string[])

    //finally, loop through the hierarchy to find the clusters that we end up selecting!
    for (let i = 0; i < hierarchy.length; i++) {
      if (isSelected[i]) {
        clusters.push(hierarchy[i].elements);
      }
    }

    // find the outliers now
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

export function euclidean(pointA: number[], pointB: number[]): number {
  if (pointA.length !== pointB.length) {
    throw new Error("unequal dimension in input data");
  }
  let sum = 0;
  for (let i = 0; i < pointA.length; i++) {
    const diff = pointA[i] - pointB[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

export function manhattan(pointA: number[], pointB: number[]): number {
  if (pointA.length !== pointB.length) {
    throw new Error("unequal dimension in input data");
  }
  let sum = 0;
  for (let i = 0; i < pointA.length; i++) {
    sum += Math.abs(pointA[i] - pointB[i]);
  }
  return sum;
}

export function cosine(pointA: number[], pointB: number[]): number {
  if (pointA.length !== pointB.length) {
    throw new Error("unequal dimension in input data");
  }
  let dotProduct = 0.0;
  let normA = 0.0;
  let normB = 0.0;
  for (let i = 0; i < pointA.length; i++) {
    dotProduct += pointA[i] * pointB[i];
    normA += pointA[i] * pointA[i];
    normB += pointB[i] * pointB[i];
  }
  if (normA === 0 || normB === 0) {
    return 1;
  }
  const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  return 1 - similarity;
}

class PriorityQueue<T> {
  private _heap: T[];
  private _comparator: (a: T, b: T) => boolean;

  constructor(comparator: (a: T, b: T) => boolean) {
    this._heap = [];
    this._comparator = comparator;
  }

  enqueue(value: T): void {
    this._heap.push(value);
    this._siftUp();
  }

  dequeue(): T | null {
    if (this.isEmpty()) {
      return null;
    }
    const poppedValue = this._heap[0];
    const bottomValue = this._heap.pop();
    if (this._heap.length > 0 && bottomValue !== undefined) {
      this._heap[0] = bottomValue;
      this._siftDown();
    }
    return poppedValue;
  }

  isEmpty(): boolean {
    return this._heap.length === 0;
  }

  size(): number {
    return this._heap.length;
  }

  peek(): T | null {
    if (this.isEmpty()) {
      return null;
    }
    return this._heap[0];
  }

  private _siftUp(): void {
    let nodeIdx = this._heap.length - 1;
    while (
      nodeIdx > 0 &&
      this._comparator(
        this._heap[nodeIdx],
        this._heap[Math.floor((nodeIdx - 1) / 2)]
      )
    ) {
      this._swap(nodeIdx, Math.floor((nodeIdx - 1) / 2));
      nodeIdx = Math.floor((nodeIdx - 1) / 2);
    }
  }

  private _siftDown(): void {
    let nodeIdx = 0;
    while (
      (2 * nodeIdx + 1 < this._heap.length &&
        this._comparator(this._heap[2 * nodeIdx + 1], this._heap[nodeIdx])) ||
      (2 * nodeIdx + 2 < this._heap.length &&
        this._comparator(this._heap[2 * nodeIdx + 2], this._heap[nodeIdx]))
    ) {
      const smallerChildIdx =
        2 * nodeIdx + 2 < this._heap.length &&
        this._comparator(
          this._heap[2 * nodeIdx + 2],
          this._heap[2 * nodeIdx + 1]
        )
          ? 2 * nodeIdx + 2
          : 2 * nodeIdx + 1;
      this._swap(nodeIdx, smallerChildIdx);
      nodeIdx = smallerChildIdx;
    }
  }

  private _swap(i: number, j: number): void {
    [this._heap[i], this._heap[j]] = [this._heap[j], this._heap[i]];
  }
}

/**
 * Finds the n most central elements in a cluster of vectors
 * @template T Type of cluster elements extending {id: string; vector: number[]}
 * @param cluster Array of objects containing at least {id, vector} properties
 * @param n Number of central elements to return (must be >= 1)
 * @param distanceFunction Optional distance function (defaults to cosine)
 * @returns Array of input objects with additional distance property, representing the n most central elements, sorted by distance from centroid
 * @throws Error if n < 1 or cluster is empty
 */
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

  // Find n closest points to centroid
  return cluster
    .map((point) => ({
      ...point,
      distance: distanceFunction(point.vector, centroid),
    }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, n);
}

//extended union-find data structure
class UnionFind<T> {
  private parent: Map<T, T>;
  private rank: Map<T, number>;

  constructor(elements: T[]) {
    this.parent = new Map();
    this.rank = new Map();

    elements.forEach((e) => {
      this.parent.set(e, e); // Each element is the parent of itself
      this.rank.set(e, 0); // Rank of each element is 0 initially
    });
  }

  find(item: T): T {
    if (this.parent.get(item)! !== item) {
      this.parent.set(item, this.find(this.parent.get(item)!));
    }
    return this.parent.get(item)!;
  }

  union(item1: T, item2: T): void {
    const root1 = this.find(item1);
    const root2 = this.find(item2);

    if (root1 === root2) return;

    const rank1 = this.rank.get(root1)!;
    const rank2 = this.rank.get(root2)!;

    if (rank1 > rank2) {
      this.parent.set(root2, root1);
    } else if (rank1 < rank2) {
      this.parent.set(root1, root2);
    } else {
      this.parent.set(root2, root1);
      this.rank.set(root1, rank1 + 1);
    }
  }
}

export default HDBSCAN;