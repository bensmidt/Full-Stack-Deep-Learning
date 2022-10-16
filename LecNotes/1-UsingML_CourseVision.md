# [When to Use ML and Course Vision](https://fullstackdeeplearning.com/course/2022/lecture-1-course-vision-and-when-to-use-ml/)

## Contents
1. [When to Use Machine Learning](#when-to-use-machine-learning)
2. [Picking Problems to Solve with ML](#picking-problems-to-solve-with-ml)
3. [Cost](#cost)
4. [Estimating Problem Difficulty](#estimating-problem-difficulty)
5. [Planning ML Projects](#planning-ml-projects)
    - [ML Archetypes](#ml-archtypes)
    - [Data Flywheel](#data-flywheel)
    - [Feasibility vs. Impact](#feasiblity-vs-impact)
    - [Tool Fetishization](#tool-fetishization)
6. [Lifecycle of ML Projects](#lifecycle-of-a-ml-project)

<p>&nbsp;</p>

# When to Use Machine Learning

- Lots of ML projects fail (~87% ??)
- ML is still research --> shouldn't aim for 100% success
- Many projects are doomed to fail before being understaken
    - Technically infeasible or poorly scoped
    - Never make the leap to production
    - Unclear success criteria
    - Works but ML doesn't solve a big enough problem

- ML introduces a lot of complexity 
    - Erodes the boundaries between systems
    - Relies on expensive data dependencies
    - Commonly plagued by system design anti-patterns
    - Subject to the instability of the external world

[Machine Learning: The High Interest of Credit Card of Technical Debt](https://research.google/pubs/pub43146/)

- Are we ready to use ML? 
- Do we really need ML to solve this problem? 
- Is it ethical? 

<p>&nbsp;</p>

# Picking Problems to Solve with ML
TL/DR: High impact, low-cost

Book: *Prediction Machines: The Simple Economics of Artificial Intelligence*

- High impact ML projects
    - AI reduces the cost of prediction
        - Used to need a person
    - Cheap predictions --> predictions everywhere
    - Look for projects where cheap prediction will have a huge business impact

[Three Principles for Designing ML-Powered Products (Spotify)](https://spotify.design/article/three-principles-for-designing-ml-powered-products)

[Software 2.0 (Karpathy)](https://karpathy.medium.com/software-2-0-a64152b37c35)
- Can replace very complex code with ML

What are other people doing? 
- [Human-centric Machine Learning Infrastructure @Netflix](https://www.infoq.com/presentations/netflix-ml-infrastructure/)
- Industry reports 
- Papers from (big papers) Google, Meta, Nvidia, Netflix
- Blogs from smaller companies (Uber, Lyft, Spotify)

<p>&nbsp;</p>

# Cost

Drivers
1. Data Availability
    - How hard is it to acquire data? 
    - How expensive is data labeling?
    - How much data will be needed? 
    - How stable is the data? 
    - Data security requirements?
2. Accuracy Requirement
    - How costly are wrong predictions?
        - Self driving = very expensive
        - Recommender system = very inexpensive (single prediction)
    - How frequently does the system need to be right to be useful? 
        - DALLE image generation just needs to be correct 1 of every n times
3. Problem Difficulty
    - Is the problem well-defined? 
    - Good published work on similar problems? 
    - Compute requirements? 
    - Can a human do it? 
        - This gives some indication that ML can solve it

Why is accuracy so important? 
- Costs tend to scale super-linearly to the accuracy requirement

<p>&nbsp;</p>

# Estimating Problem Difficulty
- This is a classically difficult problem to answer confidently 
- It's not very intuitive what computers can and can't solve
- You must be up to date on the state of the art --> ML is chaning extremely quickly 

Andrew Ng: "Pretty much anything that a normal person can do in <1 sec, we can now automate with AI"
- Not a great heuristic but works well for many examples
- Counter examples
    - Humor
    - In-hand robotic manipulation

Unsupervised learning is **NO LONGER** difficult to implement

What's hard in supervised learning? 
- Answering questions
- Summarizing text
- Prediction video
- Real-world speech recognition
- Resisting adversairal attacks

What types of problems are still difficult? 
- Output is complex
    - Instances
        - High-dimensional output
        - Ambiguous output
    - Examples
        - 3D reconstruction
        - Video prediction
        - Dialog systems
        - Open-ended recommener systems
- Reliability required
    - Failing safely out-of-distribution
    - High-precision pose estimation
- Generalization is required
    - Instances
        - Out of distribution data
        - Reasoning, planning, causality
    - Examples
        - Self-driving: edge cases
        - Self-driving: control
        - Small data

How to run ML feasbility? 
1. Are you sure you need ML at all? 
2. Put in the work up-fron to define success criteria with all of the stakeholders
3. Consider ethics of using ML
4. Do a literature review
5. Try to rapidly build a labeled benchmaark dataset
6. Build a *minimum* viable model 
7. Are you sure you need ML at all? 

<p>&nbsp;</p>

# Planning ML Projects
## ML Archtypes
1. Software 
    - Feasibility: High
    - Impact: Low
    - Taking something software already does (at leat partially) and improving it with ML
    -Examples: 
        - improve IDE code completion with ML
        - build customized recommendation system
        - better video game AI
    - Key Questions
        - Do your models truly improve performance? 
        - Does performance improvement generate business value? 
        - Do performance improvements elad to a data flywheel? 
2. Human-in-the-loop
    - Feasibility: Medium
    - Impact: Medium
    - Helping humans do their jobs better by complementing them with ML-based tools
    - Examples
        - Turn sketches into slides
        - Email auto-completion
        - Help a radiologist do their job faster
    - Key Questions
        - How good does the system need to be useful? 
        - How can you collect enough data to make it that good? 
3. Autonomous Systems
    - Feasibility: Low
    - Cost: Impact
    - Cut out humans altogether --> fully automate with ML
    - Examples
        - Full self-driving
        - Automated customer support
        - Automated website design
    - Key Questions
        - What is an acceptable failure rate for the system? 
        - How can you guarantee that it won't exceed that failure rate? 
        - How inexpensively can you label data from the system? 

<p>&nbsp;</p>

## Data Flywheel

--> more users --> more data --> better model

- Do you have a data loop? 
- Does the data actually improve your model? 
- Does your improved model actually make the product better? 

<p>&nbsp;</p>

## Feasiblity vs. Impact
1. Software 
    - Feasibility: High
    - Impact: Low
    - Improve Impact? 
        - Implement a data loop that allows you to improve on this task and future ones
2. Human-in-the-loop
    - Feasibility: Medium
    - Impact: Medium
    - Improve feasibility?
        - Good product design 
        - Release "good enough" version
3. Autonomous Systems
    - Feasibility: Low
    - Cost: Impact

JUST GET STARTED!!

<p>&nbsp;</p>

## Tool Fetishization
- Don't fall into the tool fetishization
- You don't need perfect tools
- You don't need a perfect model
- You don't need perfect infrastructure
- You don't need whatever Google or Uber is doing

MLOps at Reasonable Scale
- Computing: finite budges
- Team size: small
- Data: dataset are terabytes not petabytes

<p>&nbsp;</p>

# Lifecycle of ML Projects
Running case study: pose estimation

*You often will find yourself moving back and forth between steps*
1. Planning and Project Setup
    - Determine requirements and goals
    - Allocate resoureces
    - Consider ethical considerations
2. Data Collection and Labeling
    - Collect training objects
    - Annotate with ground truth
3. Training and Debugging
    - Implement baseline in OpenCV functions
    - Find SoTA model and reproduce
    - Debug our implementation
    - Improve model for our task
    - You may need to go back and collect more data
    - YOu may realize the task is more difficult than we thought
4. Deploying and Testing
    - Pilot in grasping system in the lab
    - Write tests to prevent regressions and evaluate for biases
    - Roll out into production
    - You'll often need to revisit the training and debugging stage
    - There are many reasons you may need to loop back to other stages, even the beginning

<p>&nbsp;</p>

# Summary
- ML is complex --> use it if it makes sense
- You don't need a perfect setup to get started







