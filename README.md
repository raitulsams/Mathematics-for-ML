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

## 2.2 Scalars and Vectors: The Dimensions of Data

**The Concept:**
In machine learning and physics, we categorize data based on how much information is needed to fully describe it. We use Scalars for simple amounts and Vectors for amounts that also have a direction or multiple features.

### Scalars: Magnitude Only

A scalar is a single number. It only has "magnitude" (a size or amount). It does not need a direction to make sense.

**Real-Life Example:** \* **Temperature:** If you say "The temperature is 30 degrees Celsius," that is a complete piece of information. You do not say "It is 30 degrees Celsius facing North."

- **Other examples:** Weight (70 kg), Age (25 years), or the price of a house ($300,000).

### Vectors: Magnitude and Direction

A vector has both magnitude (size) AND direction. In ML, you can also think of a vector as a list of distinct but related features that describe a single object.

**Real-Life Example:**

- **Distance vs. Displacement:** If you tell someone "Walk 10 meters," that is a scalar (distance). They could end up anywhere in a circle around you. If you say "Walk 10 meters East," that is a vector (displacement). You have given them a magnitude (10m) and a direction (East) to find the exact point B.

---

### Plotting Vectors and The Origin

Let's say we have a vector: `v = [1, 2]`.
If we plot this on a 2D plane (x-axis and y-axis), it looks like an arrow pointing to the coordinates where `x = 1` and `y = 2`.

**Why do vectors start from (0,0)?**
The point (0,0) is called the **Origin**. Vectors start here because we need a universal reference point to give the numbers meaning. Think of it like using a ruler: if you want to measure a 5-inch piece of wood, you have to line up the edge of the wood with the "0" mark on the ruler. If you do not start at zero, the numbers lose their context.

---

### Dimensions: 2D, 3D, and Beyond

The number of values inside the vector tells you how many dimensions it has.

- `v = [1, 2]` has two numbers. It is a 2D vector and can be drawn on a flat piece of paper.
- `v = [3, 5, 7]` has three numbers (x, y, and z axes). It cannot be described accurately on a flat 2D surface; it requires a 3D space (like a room with length, width, and height).
- **ML Context:** In Machine Learning, if an image is represented by a vector with 1,000 pixel values, we say that vector exists in a 1,000-dimensional space. We cannot visualize it, but the math works the exact same way as it does in 2D.

---

### Clarification: "Supporting Vectors" and Geometric Spaces

_Note: It is important to clarify the terminology here regarding "supporting vectors" and lines/hyperplanes._

When you plot a single vector like `v = [1, 2]`, the arrow itself is just a line segment. However, if you imagine that vector stretching out infinitely in both directions, it creates a **Line** (1D space) within the 2D graph. This is mathematically called the "span" of the vector.

**Lines, Planes, and Hyperplanes in ML:**

- **In 2D Space:** A boundary that separates data (like separating cats from dogs on a graph) is a straight **Line**.
- **In 3D Space:** A boundary that separates data becomes a flat sheet of paper, called a **Plane**.
- **In nD Space (4 or more dimensions):** A boundary that separates data is called a **Hyperplane**.

## 2.3 Row and column vectors and transpose: Organizing the Data

**The Concept:**
Just knowing the numbers inside a vector is not enough; how we arrange them physically on the page (or in the computer's memory) matters immensely for mathematical operations. Vectors can be written in two distinct ways: horizontally or vertically.

### Row Vectors vs. Column Vectors

- **Row Vector:** The numbers are written horizontally, side-by-side. It has exactly 1 row and multiple columns.
  - _Notation:_ `v = [1, 2, 3]`
  - _Real-Life Example:_ Think of a single row in an Excel spreadsheet. If you have a dataset of hospital patients, one row represents one patient's complete profile: `[Age: 45, Heart Rate: 80, Blood Pressure: 120]`.

- **Column Vector:** The numbers are written vertically, stacked on top of each other. It has multiple rows and exactly 1 column.
  - _Notation:_ `v = [ 1 ]`
    `    [ 2 ]`
    `    [ 3 ]`
  - _Real-Life Example:_ Think of a single column in that same Excel spreadsheet. If you want to isolate the "Heart Rate" data for every patient in the ward to find the average, you extract that single vertical column.
  - _ML Context:_ In higher-level linear algebra and machine learning algorithms, vectors are usually treated as **column vectors** by default unless stated otherwise.

### The Transpose Operation (T)

**The Concept:**
Transposing is a simple but critical operation. It is the act of flipping a matrix or vector over its diagonal axis. For a standalone vector, it simply means turning a Row Vector into a Column Vector, or vice versa.

- We denote this operation with a superscript "T". If your vector is named `v`, its transposed version is written as `v^T`.

**Real-Life Example: Rotating a Spreadsheet**
Imagine you print a spreadsheet where the rows are "Days of the Week" (Monday-Friday) and the columns are "Sales" and "Expenses". You realize it is formatted poorly for your presentation slide. You perform a Transpose operation: you pivot the table so the columns are now "Days of the Week" and the rows are "Sales" and "Expenses". The underlying data has not changed at all, but the shape has rotated.

- **Before Transpose (Row Vector):** `v = [100, 200, 300]`
- **After Transpose (v^T becomes a Column Vector):**
  `v^T = [ 100 ]`
  `      [ 200 ]`
  `      [ 300 ]`

**Why do we care in ML?**
When you train a model, you multiply your data (Matrices) by your model's weights (Vectors). The rules of Linear Algebra dictate that the physical dimensions of these shapes must align perfectly to be multiplied together. You cannot mathematically multiply two row vectors directly. You frequently have to transpose (`T`) one of them into a column vector so the computer can "lock" them together and compute the prediction.

## 2.4 Vectors distance from the origin: Calculating Magnitude

**The Concept:**
When we plot a vector starting from the origin (0,0), it forms an arrow pointing to a specific coordinate. The "distance from the origin" is simply the physical length of that arrow. In mathematics and physics, this length is called the **Magnitude**.

To calculate this, we use the **Pythagorean theorem** ($a^2 + b^2 = c^2$). If we know the horizontal distance (x) and the vertical distance (y), we square them, add them together, and then take the square root to find the direct, straight-line distance (c).

_Notation:_ The magnitude of a vector $v$ is typically written with double vertical bars $\|v\|$, though single bars $|v|$ are sometimes used.

### Distance in 2D Space

For a 2D vector $v = [x, y]$, the formula is:
$\|v\| = \sqrt{x^2 + y^2}$

**Real-Life Example:**
Imagine you are flying a drone. You fly it 3 meters East (x-axis) and 4 meters North (y-axis). How far is the drone from you (the origin) in a straight line?

- $v = [3, 4]$
- $\|v\| = \sqrt{3^2 + 4^2}$
- $\|v\| = \sqrt{9 + 16}$
- $\|v\| = \sqrt{25} = 5$
  The straight-line distance (magnitude) is 5 meters.

### Distance in 3D Space

When we add a third dimension (z-axis, representing depth or height), the formula elegantly scales up by simply adding the squared z-value under the square root.
For a 3D vector $v = [x, y, z]$, the formula is:
$\|v\| = \sqrt{x^2 + y^2 + z^2}$

**Real-Life Example:**
Taking the drone example further, let's say it flies 3 meters East, 4 meters North, and then straight up 12 meters into the air (z-axis).

- $v = [3, 4, 12]$
- $\|v\| = \sqrt{3^2 + 4^2 + 12^2}$
- $\|v\| = \sqrt{9 + 16 + 144}$
- $\|v\| = \sqrt{169} = 13$
  The straight-line distance from your controller to the drone in the sky is 13 meters.

### ML Context: Euclidean Distance

Why do we care about the length of an arrow in Machine Learning?

In ML, this exact formula is known as **Euclidean Distance**. It is heavily used in algorithms like K-Nearest Neighbors (KNN). If a model wants to know how "similar" two pieces of data are (e.g., determining if a new movie is similar to a movie you already like), it calculates the distance between their vectors. A shorter distance means the data points are highly similar; a longer distance means they are very different.
