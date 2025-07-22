# Beyond the Average: A Deep Dive into the Statistical Engine of AI and Vision

For many of us, the word "statistics" can conjure up images of dusty textbooks, bewildering formulas, and maybe a high school math class we’d rather forget. I get it. But this past week, I dove headfirst into some foundational statistical concepts, and what I found wasn't just theory—it was the source code for how machines perceive and interpret the world.

I realized that statistics isn't just about crunching numbers; it's the art of finding stories, patterns, and meaning in a world that's overflowing with data and noise. It’s about asking simple, intuitive questions that lead to profound answers. Questions like:

*   How do we find a single number to represent a whole group of things? (And what happens when that number lies?)
*   How can we mathematically measure "consistency" or "volatility"?
*   How do we know if two things are related? And more importantly, how do we avoid fooling ourselves?
*   How can we algorithmically change what we see just by understanding the shape of its data?

This week was a journey from these abstract questions to concrete code and visual AI applications. By the end of this, you'll see how the answer to a simple statistical question helps a computer denoise a photo, analyze texture, or find relevant information in a sea of text. 

## The Center of It All: Mean, Median, and the "Bill Gates Walks into a Bar" Problem

Our first stop is the most fundamental question of all: how do we summarize a whole pile of data with just one number? This brings us to the concept of "central tendency."

### The Mean - Our Familiar, Flawed Friend

The first tool we all learn is the **Mean**, or the arithmetic average. It's the concept we've known since grade school: you sum everything up and divide by how many there are. In statistics, the mean of a sample is considered an *estimator* of the **Expected Value** `E[X]` of the underlying probability distribution.

**Formula:**
$$
\mu = \frac{1}{N}\sum_{i=1}^{N}X_i
$$

The mean is fantastic for getting a quick sense of the "center of mass" of your data. It’s the balance point. But, as I was reminded this week, our familiar friend has a critical weakness.

#### The Outlier Dilemma: When a Billionaire Skews the Data

The mean is incredibly sensitive to **outliers**—data points that are wildly different from the rest. To understand this, let's use a classic analogy: "Bill Gates walks into a bar."

Imagine a bar with 10 patrons, each earning a respectable $50,000 a year. The mean income is, unsurprisingly, $50,000. It accurately represents the financial situation of the people there.

Now, in walks Bill Gates, with an annual income of, let's say, $1 billion. What happens to our mean income for the 11 people in the bar?

$$
\text{Mean} = \frac{(10 \times \$50,000) + \$1,000,000,000}{11} \approx \$90,954,545
$$

Suddenly, the mean income is nearly $91 million! While statistically correct, this statement is grossly misleading. The mean has been so distorted by one single outlier that it no longer tells a useful story about the typical person in the room.

### The Median - The Hero of Skewed Data

This is where our second measure, the **Median**, comes to the rescue. The median is simply the middle value when you sort all the data points in order.

Let's revisit our bar. Before Bill Gates, the middle value is $50,000. When he walks in, our new sorted list of 11 incomes is: `{$50k, $50k, ..., $50k, $1B}`. The middle value (the 6th one) is still $50,000.

The median completely ignored the outlier. It tells us about the experience of the *typical* individual, remaining "robust" even in the face of extreme values. The mean tells you the mathematical center of mass; the median tells you the story of the individual in the middle.

#### From Theory to Code

Let's see how these are implemented from scratch to understand the mechanics before using NumPy's optimized functions.

```python
import numpy as np

# A from-scratch implementation of the mean
def compute_mean(X):
  """Calculates the arithmetic mean of a list of numbers."""
  return np.sum(X) / len(X)

X_data = [2, 0, 2, 2, 7, 4, -2, 5, -1, -1]
print(f"Mean (from scratch): {compute_mean(X_data)}")
print(f"Mean (NumPy): {np.mean(X_data)}")
# Output: Mean (from scratch): 1.8
# Output: Mean (NumPy): 1.8
```

The median requires sorting and then logic to handle odd or even-sized datasets.

```python
# A from-scratch implementation of the median
def compute_median(X):
  """Calculates the median of a list of numbers."""
  size = len(X)
  X_sorted = np.sort(X) # Sort the array
  
  if (size % 2 == 0):
    # If the dataset size is even, average the two middle elements
    mid1_idx = int(size / 2) - 1
    mid2_idx = int(size / 2)
    return (X_sorted[mid1_idx] + X_sorted[mid2_idx]) / 2
  else:
    # If the dataset size is odd, return the single middle element
    return X_sorted[int((size - 1) / 2)]

X_even_data = [1, 5, 4, 4, 9, 13]
print(f"Median (from scratch): {compute_median(X_even_data)}")
print(f"Median (NumPy): {np.median(X_even_data)}")
# Output: Median (from scratch): 4.5
# Output: Median (NumPy): 4.5
```

### Application Spotlight: The Visual Difference in Image Denoising

This statistical choice has a stunningly visual impact in AI. A common problem is removing "salt-and-pepper" noise from an image—random black (0) and white (255) pixels that are extreme outliers.

A **Mean Filter** replaces each pixel with the average of its neighbors. This smudges the noise into a gray blob because the outlier's extreme value heavily skews the mean.

A **Median Filter**, however, does something magical. It replaces each pixel with the *median* of its neighbors. When the filter's window covers a noise pixel, the extreme value is pushed to the end of the sorted list and is ignored. The filter picks a more representative value from the "good" neighbors, effectively deleting the noise speck while preserving sharp edges.

This was my first big "Aha!" moment. The abstract property of "robustness to outliers" has a direct, visible, and powerful manifestation.

| Feature                 | Mean                                                              | Median                                                         |
| :---------------------- | :---------------------------------------------------------------- | :------------------------------------------------------------- |
| **Calculation**         | Sum of all values divided by the number of values.                | The middle value of a sorted dataset.                          |
| **Sensitivity to Outliers** | **High.** Extreme values can drastically skew the result.             | **Low.** It is robust and largely unaffected by outliers.       |
| **Best For...**           | Symmetrical data distributions with no significant outliers.      | Skewed data or datasets with extreme outliers.                 |
| **Image Processing Use Case**  | Image Blurring (Averaging Filter). Smooths the image but can smudge details. | Image Denoising (Median Filter). Excellent at removing "salt-and-pepper" noise. |
| **Analogy**               | The "center of mass" of the data.                                 | The "person in the middle" of the data.                         |

---

## Measuring the Vibe: Variance, Standard Deviation, and Consistency

Knowing the "center" doesn't tell the whole story. We also need to measure the **dispersion** or spread of the data.

### Variance and Standard Deviation - Quantifying Spread

*   **Variance (`var(X)` or `σ²`)** is the average of the *squared differences* from the mean. Squaring the differences ensures that negative and positive deviations don't cancel each other out and penalizes larger deviations more heavily.
    **Population Variance Formula:** $$ \text{var}(X) = \frac{1}{N}\sum_{i=1}^{N}(X_i - \mu)^2 $$

*   **Standard Deviation (`σ`)** is simply the square root of the variance. It brings the measure back into the same units as our original data and represents the "typical distance" of a data point from the mean.
    **Formula:** $$ \sigma = \sqrt{\text{var}(X)} $$

#### Population vs. Sample: The `n-1` Enigma

A crucial point is the difference between calculating variance for an entire **population** versus a **sample**. When we calculate the variance of a *sample* to *estimate* the population variance, we divide by **n-1**, not `n`.

This is **Bessel's correction**. It's needed because of **Degrees of Freedom**. When we use the *sample mean* in our variance calculation, we've used up one "degree of freedom." If you know the mean of `n` numbers, `n-1` of them can be anything, but the last one is fixed to make the mean correct. Because the sample mean is "closer" to the sample data than the true population mean, the variance calculated with `n` will, on average, be an underestimate. Dividing by `n-1` corrects for this bias.

#### From Theory to Code

Let's implement this. A from-scratch function calculates the population standard deviation (`ddof=0`).

```python
def compute_std_population(X):
  n = len(X)
  mean = np.mean(X)
  variance = np.sum((np.array(X) - mean)**2) / n
  return np.sqrt(variance)

heights = [171, 176, 155, 167, 169, 182]
print(f"Population Std Dev (from scratch): {np.round(compute_std_population(heights), 2)}")

# NumPy's np.std() by default calculates the population std dev.
print(f"Population Std Dev (NumPy, ddof=0): {np.round(np.std(heights), 2)}")

# To calculate the sample standard deviation (using n-1), we specify ddof=1
print(f"Sample Std Dev (NumPy, ddof=1): {np.round(np.std(heights, ddof=1), 2)}")
# Output:
# Population Std Dev (from scratch): 8.33
# Population Std Dev (NumPy, ddof=0): 8.33
# Sample Std Dev (NumPy, ddof=1): 9.12
```

### Application Spotlight: Teaching a Computer to "Feel" Texture

How can a computer tell a smooth sky from rough grass? By calculating the local standard deviation. In image terms, **texture is the local variation in pixel intensities**.

The process is brilliant:
1.  Slide a kernel (e.g., a 7x7 window) over every pixel in a grayscale image.
2.  At each position, calculate the standard deviation of all pixel values inside that window.
3.  The resulting value becomes the new pixel value in a "texture map."

Bright areas on this map correspond to highly textured parts, and dark areas correspond to smooth surfaces.

Here's how to implement this using `scipy`, which applies a function (`np.std`) to a sliding window:

```python
import cv2
import numpy as np
from scipy.ndimage.filters import generic_filter

# Load an image and convert to grayscale
img = cv2.imread('img.jpg') # Assuming a church image 'img.jpg' is available
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert to float for calculation
x = gray.astype('float')

# Apply a generic_filter. For each 7x7 window, compute the standard deviation.
# This creates the texture map.
x_filt = generic_filter(x, np.std, size=7)

# We can then enhance this map for better visualization
# For example, thresholding to remove low-level noise
x_filt[x_filt < 20] = 0 
# And scaling to use the full dynamic range for better visibility
x_filt = x_filt * 2.5 
cv2.imwrite('edge_s4.jpg', x_filt) # Save the resulting texture map
```
before:

<img width="521" height="295" alt="image" src="https://github.com/user-attachments/assets/79db8ddc-69e0-4d54-a5ed-d6b62c0d9927" />

after:

<img width="527" height="301" alt="image" src="https://github.com/user-attachments/assets/7d795b3d-9a2c-4b0b-a156-b143e72a72b5" />

The abstract concept of "dispersion" becomes the computer's direct method for "perceiving" physical texture.

---

## The Art of Connection: Covariance, Correlation, and Not Getting Fooled

Real-world data is about relationships. To quantify them, we use covariance and correlation.

### From Covariance to Correlation

**Covariance** measures the joint variability of two variables. A positive covariance indicates they move in the same direction; negative indicates opposite directions. The problem? Its magnitude is unscaled and hard to interpret.

**Pearson Correlation Coefficient (ρ or r)** solves this by normalizing the covariance. It divides the covariance by the product of the standard deviations of the two variables, scaling the result to a universal range of **-1 to +1**.

**Formula:**
$$
\rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2} \sqrt{\sum (y_i - \bar{y})^2}}
$$

| Value of r | Interpretation               |
| :--------- | :--------------------------- |
| `+1.0`       | Perfect Positive Correlation |
| `0`        | No Linear Correlation        |
| `-1.0`       | Perfect Negative Correlation |

### The Most Important Lesson: "Correlation is Not Causation"

This is the golden rule of statistics. The classic example is the strong positive correlation between **ice cream sales and shark attacks**. The hidden factor, or **lurking variable**, is hot weather, which independently causes increases in both.

---

### Application Spotlight: Shaping Reality with Histogram Equalization

Our final stop explores the very "shape" of data and how we can manipulate it to change what we see.

Have you ever had a photo that looked gray and washed-out, where all the details seem to be lost in a murky haze? **Histogram Equalization** is a powerful technique that can automatically and dramatically improve its contrast, often with a single line of code.

#### The Fingerprint of an Image: Understanding Histograms

First, we need to understand what an image histogram is. A **Probability Mass Function (PMF)** gives the probability for each outcome of a discrete variable (like a dice roll). An **image histogram** is a simple bar chart that acts as the image's PMF. It shows the frequency of each pixel intensity value, from pure black (0) to pure white (255). The histogram is like a fingerprint of an image's tonal properties:

*   A **dark image** will have a histogram bunched up on the left side (low-intensity values).
*   A **bright image** will have one bunched up on the right (high-intensity values).
*   A **low-contrast, washed-out image** will have its histogram clumped into a narrow band somewhere in the middle. This means it's not using the full available range of black to white, which is why it looks dull.

#### The Cumulative Story: Understanding the CDF

If the histogram is the PMF, then the **Cumulative Distribution Function (CDF)** is its running total. For any intensity level *k*, the CDF tells you the proportion of pixels in the image that have an intensity of *k* or less. For a low-contrast image, its CDF will be mostly flat, then rise very steeply over that narrow band of intensities, and then become flat again at the top. This steep rise is the statistical signature of low contrast.

#### The Magic: Stretching the Histogram

The core idea of Histogram Equalization is simple and brilliant. **Analogy:** Imagine all the pixels in our low-contrast image are people standing in a line that goes from 0 to 255. Because the contrast is low, they're all crowded together in a small section, say from 100 to 150. Histogram equalization is like a teacher telling them, "Spread out! Use the entire length of the line evenly."

The "how" is the magical part. The CDF itself provides the mathematical formula that tells each pixel where it needs to move to achieve this "spreading out."

**Formula:**
$$
\text{new\_pixel\_value} = \text{round}((L-1) \times \text{CDF}(\text{old\_pixel\_value}))
$$
Where `L` is the number of gray levels (typically 256 for an 8-bit image). This process takes the pixels in the densely populated part of the histogram (where the CDF is rising steeply) and stretches them far apart. The result is an image with a much more uniform, "flat" histogram, which corresponds visually to a high-contrast image.

#### A Practical Demonstration with Python

Let's make this real. Here is a complete script that you can run on your own low-contrast image. For this demonstration, I'll assume you have an image named `low_contrast.jpg` in the same directory.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Load the Image and Convert to Grayscale ---
# We'll work with a local image file. Make sure 'low_contrast.jpg' exists.
try:
    image = cv2.imread('low_contrast.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
except cv2.error as e:
    print("Error: Could not load image. Make sure 'low_contrast.jpg' is in the correct path.")
    # As a fallback for demonstration, create a synthetic low-contrast image
    base = np.arange(0, 256).astype('uint8')
    low_contrast_row = np.clip(base * 0.5 + 80, 0, 255).astype('uint8')
    gray_image = np.tile(low_contrast_row, (256, 1))

# --- Step 2: Perform Histogram Equalization ---
# This single OpenCV function performs all the PMF, CDF, and mapping steps internally.
equalized_image = cv2.equalizeHist(gray_image)

# --- Step 3: Calculate Histograms for Visualization ---
# We calculate the histograms to see the "before" and "after" distribution.
# cv2.calcHist([image], [channel], mask, [histSize], [ranges])
hist_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

# --- Step 4: Visualize the Results ---
# Set up a figure to display everything
plt.figure(figsize=(15, 12))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Low-Contrast Image')
plt.axis('off')

# Histogram of Original Image
plt.subplot(2, 2, 2)
plt.plot(hist_original, color='b')
plt.title('Histogram of Original Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])

# Equalized Image
plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Histogram Equalized Image')
plt.axis('off')

# Histogram of Equalized Image
plt.subplot(2, 2, 4)
plt.plot(hist_equalized, color='r')
plt.title('Histogram of Equalized Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()
```
<img width="1641" height="1095" alt="image" src="https://github.com/user-attachments/assets/88c802ef-22c2-4cf0-929b-da4011bf454b" />

**Analyzing the Output:**
Theoretically, the new image should be expected to have deep blacks, bright whites and much more visible detail in the mid tones. HOWEVER, **In many cases, especially for artistic or natural photographs, standard histogram equalization *does* make the image look "worse" or at least less natural.**And this is exactly the case with this result. It reveals the crucial difference between a purely mathematical optimization and an aesthetically pleasing result.

The code is working exactly as intended, and the output we're seeing perfectly demonstrates the strengths and, more importantly, the significant weaknesses of this algorithm. Let's break down exactly what's happening.

### Analyzing the Output: Why Does It Look "Worse"?

Our original image, while having low contrast, has a soft, atmospheric quality. The light rays are subtle, and the fog in the valley has smooth gradations. The equalized image, on the other hand, looks harsh, grainy, and almost "posterized" in the sky.

Here’s why this happens, connecting it directly to the histograms you've plotted:

**1. The "Global" Nature of the Algorithm**

Standard `cv2.equalizeHist()` is a **global** operation. It calculates a *single* histogram for the *entire* image and then applies *one single* mapping function (derived from the CDF) to *every pixel*. It's a blunt instrument.

Look at your original histogram (the blue one). There is a massive "hump" of pixels concentrated in the mid-gray range (roughly between intensities 40 and 100). These are the pixels that make up the vast areas of sky, clouds, and mist.

Because this region is so dominant, it completely dictates the shape of the CDF. The CDF will rise *extremely* steeply in this specific range.

**2. Creation of "Banding" and Harsh Transitions**

According to the equalization formula, a steep rise in the CDF means that even small differences in the original input gray levels will be stretched out dramatically into large differences in the output.

*   Imagine original pixel values `50`, `51`, and `52` are all in the middle of that steep CDF curve.
*   The equalization might map them to new values like `60`, `75`, and `90`.
*   A subtle 2-level difference has been amplified into a harsh 30-level difference.

This is exactly what creates the harsh **"banding"** or **"posterization"** effect you see in the sky and sunbeams. The algorithm is aggressively trying to spread out that huge clump of mid-tones, destroying the subtle, smooth gradations that made the original look natural.

**3. Amplification of Noise**

This is the second major side effect. In the relatively smooth areas of the original sky, there's always a little bit of imperceptible noise (e.g., pixel values might be `120`, `121`, `120`, `122`). These are tiny, insignificant differences.

However, because these values also fall within that steep part of the CDF curve, the algorithm grabs these tiny differences and stretches them into much larger, now *visible* differences. This is why the smooth areas in the original become grainy and noisy in the equalized version. The algorithm is "enhancing" imperceptible noise just as much as it's enhancing real features.

### So, When is Histogram Equalization Actually Useful?

This leads to a critical conclusion: global histogram equalization is a powerful tool, but it's not a universal "make my photo look better" button.

| Histogram Equalization is GOOD for:                                                                                   | Histogram Equalization is BAD for:                                                                                    |
| :-------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| **Objective, Scientific Analysis:** Enhancing contrast in medical images (X-rays) or satellite photos where revealing underlying structures is more important than aesthetics. | **General Photography:** Where artistic intent, natural gradations, and subtle lighting are important.                |
| **As a Pre-processing Step:** For other computer vision algorithms (like feature detection) that perform better on high-contrast images. | **Images with already good contrast:** It can create an overly harsh and artificial look.                             |
| **Severely Under or Over-Exposed Images:** When an image is almost entirely black or entirely white, it can sometimes recover lost details. | **Images with large, uniform regions (like a clear sky):** It will drastically amplify any noise present in those areas. |

### The Better Solution: Going Local with CLAHE

So how do we get the benefits of contrast enhancement without these nasty side effects? The solution is to stop thinking globally and start thinking **locally**.

This brings us to **CLAHE (Contrast Limited Adaptive Histogram Equalization)**. It's a much smarter algorithm that solves both of the problems we identified.

1.  **Adaptive:** Instead of using one histogram for the whole image, CLAHE divides the image into small blocks (e.g., 8x8 tiles) and applies histogram equalization to each tile *independently*. This allows it to adapt to the local context. It can enhance the contrast in the dark forest area based on the forest's pixels, and separately enhance the sky based on the sky's pixels, leading to a much more natural result.

2.  **Contrast Limited:** To prevent the noise-amplification problem, CLAHE introduces a "contrast limit." Before calculating the CDF for a tile, it "clips" the histogram. If a certain intensity value is over-represented (a high spike, which is often caused by noise in a uniform area), it cuts off the spike at a predefined limit and redistributes that excess count among all the other bins. This prevents a single intensity from dominating the transformation and excessively amplifying noise.

Here is how you would implement it. Notice it's just as easy as the original function call in OpenCV:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Load the Image and Convert to Grayscale (Same as before) ---
try:
    image = cv2.imread('low_contrast.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
except cv2.error as e:
    print("Error: Could not load image. Make sure 'low_contrast.jpg' is in the correct path.")
    # Fallback image
    base = np.arange(0, 256).astype('uint8')
    low_contrast_row = np.clip(base * 0.5 + 80, 0, 255).astype('uint8')
    gray_image = np.tile(low_contrast_row, (256, 1))

# --- Step 2: Perform STANDARD Histogram Equalization (for comparison) ---
equalized_image = cv2.equalizeHist(gray_image)

# --- Step 3: Perform CLAHE ---
# Create a CLAHE object. Two important parameters:
# clipLimit: The threshold for contrast limiting.
# tileGridSize: The size of the grid for adaptive equalization.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(gray_image)

# --- Step 4: Visualize Everything ---
plt.figure(figsize=(18, 10))

# Original Image and its Histogram
plt.subplot(2, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.plot(cv2.calcHist([gray_image], [0], None, [256], [0, 256]))
plt.title('Original Histogram')
plt.xlim([0, 256])

# Standard Equalized Image and its Histogram
plt.subplot(2, 3, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Standard Global Equalization')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.plot(cv2.calcHist([equalized_image], [0], None, [256], [0, 256]))
plt.title('Global Equalized Histogram')
plt.xlim([0, 256])

# CLAHE Image and its Histogram
plt.subplot(2, 3, 3)
plt.imshow(clahe_image, cmap='gray')
plt.title('CLAHE (Adaptive Equalization)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.plot(cv2.calcHist([clahe_image], [0], None, [256], [0, 256]))
plt.title('CLAHE Histogram')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()
```
When you run this, you will see that the CLAHE result is vastly superior. It enhances local contrast in the trees and the hills while preserving the smooth texture of the sky, avoiding the noise amplification and harshness of the global method.

<img width="1919" height="1094" alt="image" src="https://github.com/user-attachments/assets/0c3b3dc4-1c4d-436f-9b34-10bd77f7dfa1" />

**Conclusion:** Your observation was spot on. It highlights that understanding the *limitations* of an algorithm is just as important as knowing how to use it. Global histogram equalization is a powerful but blunt tool, and recognizing when it fails leads us directly to more sophisticated and effective techniques like CLAHE.

#### A Manual Walkthrough

To see *exactly* how the CDF mapping works on a micro-level, let's trace the algorithm on the tiny 5x5 image from the notebook, where pixel values range from 0 to 7 (`L=8`).

```
Simple Image (5x5):
[[1, 2, 7, 5, 6],
 [7, 2, 3, 4, 5],
 [0, 1, 5, 7, 3],
 [1, 2, 5, 6, 7],
 [6, 1, 0, 3, 4]]
```

**Step 1: Calculate PMF (Frequency / Total Pixels)**
First, we count the occurrences of each pixel value. Total pixels = 25.

| Intensity | Frequency | PMF (Freq / 25) |
| :-------- | :-------- | :-------------- |
| 0         | 2         | 2/25 = 0.08     |
| 1         | 4         | 4/25 = 0.16     |
| 2         | 3         | 3/25 = 0.12     |
| 3         | 3         | 3/25 = 0.12     |
| 4         | 2         | 2/25 = 0.08     |
| 5         | 4         | 4/25 = 0.16     |
| 6         | 3         | 3/25 = 0.12     |
| 7         | 4         | 4/25 = 0.16     |

**Step 2: Calculate CDF (Cumulative Sum of PMF)**
Now, we compute the running total of the PMF.

| Intensity | PMF   | CDF   |
| :-------- | :---- | :---- |
| 0         | 0.08  | 0.08  |
| 1         | 0.16  | 0.24  |
| 2         | 0.12  | 0.36  |
| 3         | 0.12  | 0.48  |
| 4         | 0.08  | 0.56  |
| 5         | 0.16  | 0.72  |
| 6         | 0.12  | 0.84  |
| 7         | 0.16  | 1.00  |

**Step 3: Map New Values using the CDF**
We apply the formula `round((L-1) * CDF) = round(7 * CDF)` to each intensity level.

| Intensity (Old) | CDF   | `7 * CDF` | New Value |
| :-------------- | :---- | :-------- | :-------- |
| 0               | 0.08  | 0.56      | 1         |
| 1               | 0.24  | 1.68      | 2         |
| 2               | 0.36  | 2.52      | 3         |
| 3               | 0.48  | 3.36      | 3         |
| 4               | 0.56  | 3.92      | 4         |
| 5               | 0.72  | 5.04      | 5         |
| 6               | 0.84  | 5.88      | 6         |
| 7               | 1.00  | 7.00      | 7         |

**Step 4: Create the New Image**
Finally, we replace each pixel in the original image with its new mapped value. For instance, every pixel that was `0` becomes `1`, every pixel that was `1` becomes `2`, and every pixel that was `3` also becomes `3`. This re-mapping stretches the pixel values to better utilize the available intensity range.

---

## Conclusion & Final Thoughts

We've journeyed from the simple idea of an average all the way to reshaping the visual fabric of an image. We saw how the mean's vulnerability to outliers is mirrored in a blurry filter, while the median's robustness gives us a powerful denoising tool. We learned that standard deviation isn't just a measure of risk, but a way for a computer to perceive texture. We uncovered how correlation can find a face in a crowd, and how the very distribution of an image's data holds the key to enhancing its own contrast.

These concepts are more than just tools; they are fundamental building blocks. Understanding them deeply is the first step toward grasping much more advanced topics in machine learning and AI. They prove that sometimes, the most powerful ideas are the most fundamental ones.
