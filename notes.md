### Question: Do demographics, health history (past 24 months) and treatment status (classic vs classic+) predict current depression severity?

## Why is this useful to ask?
It is useful to know (a) whether demographics and/or (b) health factors and/or their interaction with treatment type predict depression severity because this could allow for valuable inferences regarding patient characterstics that make it more or less likely a given patient will respond to treatment.

## Limitations
Due to the limited time available, there are a number of limitations that need to be kept in mind.
- Data cleaning and preprocessing: Typically I would check the following (as the bare minimum)
    - Missing values (type of missing? eg. missing completely at random etc)
    - Duplicates
    - Basic data type validation (expected vs actual)
- Data exploration:
    - plot potential predictors against outcome
- Model selection: explore different types of models (linear, non-linear, simple to complex)

## Outcome (y)
We do not have access to a measure of current diagnosis status, but we have proxy measures in the form of binary variables (current anhedonia, current depressive mood, current suicidal thoughts).
We can use these as proxies to construct a (crude) measure of depression severity. If a patient ladders up to the suicidal thoughts stage, he/she will be recorded as high severity, whereas a score of up to depressed mood and anhedonia will be recorded as moderate and mild, respectively. There are obvious caveats with this approach - these measures are not rooted in clinical diagnosis guidelines and represent a 'best we can do with what is available' type of appraoch. Ideally I would look at the literature to confirm that this is a reasonable approach to take and/or consult an expert, but time limitations did not allow for that in this case.

### compound measures - predictors
- medical history - physical: diabetes, high BP, renal failure - presence of each of these to be summed, with a maximum score of 3 and minimum of 0, 3 indicating higher comorbidity levels
- medical history - mental health: "classic" vs "classic+" - ie any ad + therapy or ad + therapy + other medication - this is crude but gives the basic idea (who benefits from what)
- treatement type (antidepressant): if trt_adt or trt_ssr == 1, set adt to 1