/**
 * A simple Matrix class with basic operations like multiplication, addition, subtraction, transposition, and Hadamard product.
 * Each method is static and operates on 2D arrays representing matrices.
 */
export class Matrix {
  static multiply(a, b) {
    let result = new Array(a.length).fill(0).map(() => new Array(b[0].length).fill(0));
    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < b[0].length; j++) {
        for (let k = 0; k < a[0].length; k++) {
          result[i][j] += a[i][k] * b[k][j];
        }
      }
    }
    return result;
  }

  static subtract(a, b) {
    let result = new Array(a.length).fill(0).map(() => new Array(a[0].length).fill(0));
    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < a[i].length; j++) {
        result[i][j] = a[i][j] - b[i][j];
      }
    }
    return result;
  }

  static add(a, b) {
    if (a.length !== b.length || a[0].length !== b[0].length) {
      console.error("Matrices must have the same dimensions for addition.");
      return null;
    }

    let result = new Array(a.length).fill(0).map(() => new Array(a[0].length).fill(0));
    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < a[0].length; j++) {
        result[i][j] = a[i][j] + b[i][j];
      }
    }
    return result;
  }

  static transpose(matrix) {
    let result = new Array(matrix[0].length).fill(0).map(() => new Array(matrix.length).fill(0));
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[0].length; j++) {
        result[j][i] = matrix[i][j];
      }
    }
    return result;
  }

  static hadamard(a, b) {
    if (a.length !== b.length || a[0].length !== b[0].length) {
      console.error("Matrices must have the same dimensions for Hadamard product.");
      return null;
    }

    let result = new Array(a.length).fill(0).map(() => new Array(a[0].length).fill(0));
    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < a[0].length; j++) {
        result[i][j] = a[i][j] * b[i][j];
      }
    }
    return result;
  }

  static random(rows, cols) {
    return new Array(rows).fill(0).map(() => new Array(cols).fill(0).map(() => Math.random() * 2 - 1));
  }

  static fromArray(arr) {
    // Input: [1, 2, 3]
    // Output: [[1], [2], [3]] (3x1 Matrix)
    return arr.map((value) => [value]);
  }

  static toArray(matrix) {
    // Input: [[1], [2], [3]] (3x1 Matrix)
    // Output: [1, 2, 3]
    let arr = [];
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[0].length; j++) {
        arr.push(matrix[i][j]);
      }
    }
    return arr;
  }
}
