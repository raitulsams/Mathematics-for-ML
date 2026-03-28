# Mathematics for Machine Learning: Study Notes

## 1.0 Intro: The Big Picture - How Math Fits into ML

Machine Learning is essentially building a mathematical function that finds patterns in data.

**The ML Pipeline:**
`Raw Data` -> `Convert to Numbers` -> `Train Model (Find Patterns)` -> `Predict on Unseen Data`

Because computers only understand numbers, everything must be converted into numerical formats (Matrices and Vectors).

- **Text:** Converted into numerical data (word counts or embeddings).
- **Images:** Converted into numerical data (pixel intensity grids).
- **Data Representation:** Done via Linear Algebra.

**The Core Math Pillars:**

- **Statistics:** Used to understand data distributions, linear tendencies (unbiased values), spread, and outliers.
- **Linear Algebra:** Models find patterns through matrix and vector operations.
- **Calculus:** Models make predictions and there might be errors. Calculus is used to focus on optimizing and minimizing these errors.
- **Probability:** Handles uncertainty (e.g., "There is a 10% probability this is a dog, and 90% it's a cat. It's a cat."). Naive Bayes theorem is a classic example.

---

## 1.1 Equation and types of equation: The Building Blocks of Models

An equation is a mathematical way to solve a problem based on a given sentence or relationship.
_Example:_ A ball's price is twice the bat's price plus 5 extra, and the total is 25. Let the bat's price be x. The equation is `2x + 5 = 25`.

- **Linear Equations:** The degree of the variable is 1.
  - _Equation:_ `y = 3x - 1`
  - _Graph:_ A straight line. The slope is constant, with no curves.
  - _ML Context:_ We primarily start with linear equations in ML.
- **Non-Linear Equations:** The degree of the variable is more than 1.
  - _Equation:_ `y = x^2 - 6`
  - _Graph:_ A curve. The slope is not constant, which causes the curve.

---

## 1.2 Plotting an equation with 2 variables in graph: Visualizing Linear Equations

Let's create an equation from a scenario: `x (bat's price) + y (ball's price) = 25`.
We make equations so that we can solve for the value of specific items.

If we only have one equation (one piece of information) for two unknown variables, we cannot solve the problem for a single unique answer. However, we can take any value for x, calculate the corresponding y, and plot it on a 2D graph.

This creates a straight line. Every single point on that line satisfies the equation (e.g., if we pick a point from the line where x = 10.13 and y = 14.87, the total value is still 25).

---

## 1.3 Solving a linear equation: Intersecting, Overlapping, and Parallel Lines

When we have multiple unknown variables, we need multiple equations to find a solution.

**1. Unique Solution (Intersecting Lines)**

- `x + y = 25`
- `x + 2y = 30`
- We have two equations and two distinct pieces of information. Solving this gives `x = 20, y = 5`. There is only one solution. If plotted, the two lines intersect at exactly one point.

**2. Multiple/Infinite Solutions (Overlapping Lines)**

- `x + y = 25`
- `2x + 2y = 50`
- We have the exact same information from both equations (the second is just the first multiplied by 2). Thus, we will have multiple solutions. If plotted, the two lines overlap each other completely.

**3. No Solution (Parallel Lines)**

- `x + y = 25`
- `x + y = 26`
- The two pieces of information contradict each other. Thus, there will be no solution. If plotted, the two lines are parallel and have no intersection.

---

## 1.4 Plotting an eq with 3 variables: Moving to 3D Space

_Equation:_ `3x + 2y + z = 8`
Instead of a line on a 2D flat graph, an equation with 3 variables represents a flat plane in a 3D space.

---

## 1.5 Equation of a line fixed: Standard Form and Intercepts

Linear equations like `2x + 5y = 30` or `3x + y = 9` both represent lines.

- **Standard Form:** `ax + by + c = 0`
- **y-intercept Form:** `y = ax + c` (or `y = mx + c`)
  - Example: `y = -3x + 9`
  - `-3` is the slope (how the value of y changes if we change the x-axis by one unit).
  - `9` is the y-axis interception (acting as the bias).

---

## 1.6 Slope: Weights and Biases

- **Traditional Math:** `y = mx + c`
- **Machine Learning:** `y = wx + b`
  - `w` (Weight) replaces `m` (Slope).
  - `b` (Bias) replaces `c` (y-intercept).

---

## 1.7 Connecting the concepts to ML: Linear Regression in Action

Let's connect `y = mx + c` directly to Machine Learning using a student's study hours and their marks.
`Data -> Model (finds pattern) -> Prediction on unseen data`

**Training Data:**

- 1 hr -> 5 marks
- 2 hrs -> 7 marks
- 4 hrs -> 10 marks

**Prediction Goal:** 3 hrs -> marks?

We plot the hours (x-axis) and marks (y-axis) on a graph. The model creates a linear equation (a line) that fits these points with the minimum difference/error.

- `Marks = (Weight * Hours) + Bias`
- `y' = wx + b`

**Adding Features (Multiple Variables):**

- 1 Feature: `y' = w1*x1 + b` (Predicting marks based only on study hours)
- 2 Features: `y' = w1*x1 + w2*x2 + b` (Adding a feature like hours played, which impacts marks)
- 3 Features: `y' = w1*x1 + w2*x2 + w3*x3 + b` (Adding a feature like the shoes a student wears, which does not impact marks; the model will assign this a weight near 0).

_Visualizing Dimensions in ML Predictions:_

- 2D Graph = Line
- 3D Graph = Plane
- nD Graph = Hyperplane

---

## 2.1 Basic understanding of Vectors in ML: The Language of Data

A vector is simply a list of numbers. In ML, a vector represents a single row of data (one data point) and all of its features. This allows computers to process complex data efficiently using Linear Algebra rather than dealing with individual variables one by one.
