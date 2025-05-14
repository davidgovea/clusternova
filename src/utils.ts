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

export class PriorityQueue<T> {
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

export class UnionFind<T> {
  private parent: Map<T, T>;
  private rank: Map<T, number>;

  constructor(elements: T[]) {
    this.parent = new Map();
    this.rank = new Map();

    elements.forEach((e) => {
      this.parent.set(e, e);
      this.rank.set(e, 0);
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
