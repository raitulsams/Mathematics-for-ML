# Mathematics for Machine Learning: Detailed Study Notes

## 1.0 Intro: How Math Powers Machine Learning

**The Concept:** Computers are essentially giant calculators; they do not understand concepts like "cats," "sadness," or "English." They only understand numbers. Machine Learning is the process of turning real-world data into numbers, feeding those numbers into mathematical equations (the Model), and adjusting those equations until they can accurately predict outcomes.

**Real-Life Example: The Watermelon Test**
Imagine you are trying to pick the perfect, ripe watermelon at the grocery store.

- **Data:** You tap the watermelon (audio) and look at its stripes (visual).
- **Conversion:** Your brain subconsciously turns this into data points: Pitch of the tap = 4 (hollow), Darkness of stripes = 8 (dark).
- **Model:** Your brain finds a pattern: `Hollow tap + Dark stripes = Ripe`.
- **Prediction:** You tap a new, unseen watermelon. It sounds hollow and looks dark. You predict it is ripe and buy it.

**The 4 Math Pillars in ML:**

1. **Linear Algebra:** Translates your data (like the watermelon's sound and color) into grids of numbers (Matrices) so the computer can process them.
2. **Statistics:** Helps you understand your data (e.g., "Most ripe watermelons weigh around 10 lbs"). It measures linear tendency (values that are not biased), spread, and outliers.
3. **Calculus:** The engine of learning. When your model makes a bad prediction, Calculus is used to focus on optimizing and minimizing the error.
4. **Probability:** Handles uncertainty. For example, an image classification model doesn't just say "cat"; it says "There is a 90% probability this is a cat, and a 10% probability it is a dog." (Naive Bayes theorem is a classic example).

---

## 1.1 Equation and types of equation: The Rules of the Game

**The Concept:**
An equation is a mathematical sentence stating that two things are equal. We use them to solve for unknowns based on a given scenario.
_Example:_ A ball's price is twice the bat's price plus 5 extra, and the total is 25. Let the bat's price be x. The equation is `2x + 5 = 25`.

- **Linear Equations:** Variables have a degree of 1 (like x, not x^2). They create straight lines with a constant, unchanging slope.
  - _Real-Life Example:_ You rent a scooter. It costs $2 just to unlock it, plus $0.50 per minute of riding. The equation is `Total Cost = 0.50 * (minutes) + 2`. The cost goes up at a steady, straight rate. Machine learning relies heavily on linear equations because they are simple and fast to compute.
- **Non-Linear Equations:** Variables have a degree greater than 1 (like x^2). They create curved lines because the slope is not constant; the rate of change is constantly accelerating or decelerating.
  - _Real-Life Example:_ Dropping a bowling ball off a roof. It falls faster and faster the closer it gets to the ground.

---

## 1.2 Plotting an equation with 2 variables in graph: Visualizing the Data

**The Concept:**
Let's create an equation from a scenario: `x (bat's price) + y (ball's price) = 25`.

If we only have one equation (one piece of information) for two unknown variables, it is a puzzle we cannot solve for a single unique answer. However, we can take any value for x, calculate the corresponding y, and plot it on a 2D graph.

This creates a straight line. Every single point on that line satisfies the equation (e.g., if we pick a point from the line where x = 10.13 and y = 14.87, the total value is still 25).

---

## 1.3 Solving a linear equation: Finding the Truth

When we have multiple unknown variables, we need multiple pieces of information (multiple equations) to find a solution. How these lines interact on a graph tells us everything about the solution.

**1. Unique Solution (Intersecting Lines)**

- `x + y = 25`
- `x + 2y = 30`
- _Meaning:_ We have two distinct facts. If you map them, the lines cross at exactly one point (`x = 20, y = 5`). You have solved the puzzle.

**2. Infinite Solutions (Overlapping Lines)**

- `x + y = 25`
- `2x + 2y = 50`
- _Meaning:_ The second equation is just the first one multiplied by 2. We have the exact same information from both equations. The lines sit right on top of each other, meaning there are multiple (infinite) valid solutions.

**3. No Solution (Parallel Lines)**

- `x + y = 25`
- `x + y = 26`
- _Meaning:_ The two pieces of information contradict each other. Two numbers cannot add up to both 25 and 26. If plotted, the lines run perfectly parallel and will never intersect.

---

## 1.4 Plotting an eq with 3 variables: Moving to 3D Space

_Equation:_ `3x + 2y + z = 8`
Instead of a line on a 2D flat graph, an equation with 3 variables represents a flat plane in a 3D space.

---

## 1.5 & 1.6 Equations of a Line, Slopes, and ML Terminology

**The Concept:**
Linear equations like `2x + 5y = 30` or `3x + y = 9` both represent lines. They can be represented in Standard Form (`ax + by + c = 0`) or y-intercept form (`y = mx + c`).

- **Traditional Math:** `y = mx + c`
  - **m (Slope):** The angle of the line. How the value of y changes if we change the x-axis by one unit.
  - **c (y-intercept):** Where the line starts on the vertical axis when x is zero.

- **Machine Learning:** `y = wx + b`
  - **w (Weight):** Replaces the slope. It represents how _important_ or impactful a feature is.
  - **b (Bias):** Replaces the y-intercept. It represents the baseline assumption before any data is considered.

---

## 1.7 Connecting the concepts to ML: Linear Regression in Action

**The Concept:**
In ML, we do not know the exact equation right away. We start with the raw data, plot it on a graph as dots, and ask the computer to draw a line that gets as close to all the dots as possible (minimizing the difference/error).

**Real-Life Example: Study Hours vs. Marks**

- **Data:** 1 hr = 5 marks, 2 hrs = 7 marks, 4 hrs = 10 marks.
- **Goal:** Predict marks for 3 hours of study.
- **Training:** We plot the hours (x-axis) and marks (y-axis). The model drops a line onto the graph and adjusts the Weight and Bias until the total error is as small as possible.
- **Equation:** `Marks = (Weight * Hours) + Bias` (`y' = wx + b`)

**Adding Features (Multiple Variables):**
Real life isn't based on just one variable. What if we have more data?

- **1 Feature (2D Line):** `y' = w1*x1 + b` (Predicting marks based only on study hours).
- **2 Features (3D Plane):** `y' = w1*x1 + w2*x2 + b` (Adding a feature like hours spent playing, which impacts marks).
- **3+ Features (nD Hyperplane):** `y' = w1*x1 + w2*x2 + w3*x3 + b` (Adding a feature like the shoes a student wears. This does not impact marks, so the model will learn to assign it a Weight (w3) of near 0, effectively ignoring it).

---

## 2.1 Basic understanding of Vectors in ML: The Language of Data

**The Concept:**
A vector is simply a list of numbers. In ML, a vector represents a single row of data (one data point) and all of its features. Instead of writing out massive, clunky equations for every single variable, we group these features into a Vector.

**Real-Life Example:**
If we want to represent a student to our ML model, we do not feed it variables one by one. We hand it a single Vector containing all their features:
`Student A = [Study Hours: 4, Sleep Hours: 7, Classes Attended: 12]`

Mathematically, this is written simply as `v = [4, 7, 12]`. Vectors allow computers to multiply and process thousands of data points simultaneously using Linear Algebra, which is why ML models can train so quickly.
