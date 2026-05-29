# University of Technology Sydney
## 41082 - Introduction to Data Engineering

# Weather Effects on NSW Household Energy Consumption

*Assessment 2 - Experimental Design Report*

**Project Team 13:** Thomas Banzet, Fergus Hudson, Hugh Judd, Kaylan Hafza, Rhys Wuhrer, Taha Shah

*Autumn 2026*

---

## Contents

1. Analysis of the Problem
   - 1.1 Growing Importance of Electricity Demand Forecasting
   - 1.2 Project Objectives and Research Questions
   - 1.3 Why New South Wales Is a Relevant Case Study
   - 1.4 Infrastructure, Economic, and Environmental Impacts
   - 1.5 Renewable Energy and Weather Dependency
   - 1.6 Role of Publicly Available Datasets
   - 1.7 Societal Relevance and Expected Outcomes
2. Description of the Hypotheses Being Tested and the Model Being Produced
   - 2.1 Context
   - 2.2 Hypotheses
     - 2.2.1 Primary Hypothesis: Temperature
     - 2.2.2 Secondary Hypotheses: Remaining Weather Variables
     - 2.2.3 Structural Hypothesis: Climate Zone Comparison
   - 2.3 Model Description
3. Collection of the Data
   - 3.1 Collecting Data via Public Datasets
   - 3.2 Dependent and Independent Variables in the Study
   - 3.3 Confounding Variables and How to Handle Them
4. Techniques Selected
   - 4.1 Exploratory Analysis
   - 4.2 Hypothesis Tests
   - 4.3 Regression Modeling
   - 4.4 Random Forest Model
5. Discussion on the Ethics of the Problem
   - 5.1 Consent
   - 5.2 Clarity
   - 5.3 Consistency
   - 5.4 Control
   - 5.5 Consequences

---

## Introduction

This document is our Experimental Design Report (Assessment task 2) on a data engineering project. It contains the plan for our project on the impact of weather on the household energy consumption in NSW. The table below contains the contribution of each group member for this assessment task.

| Group Member | Contribution |
|---|---|
| Fergus Hudson | Section 2 |
| Hugh Judd | Section 1 |
| Kaylan Hafza | Section 3.1 and Conclusion |
| Rhys Wuhrer | Section 5 |
| Taha Shah | Section 4 |
| Thomas Banzet | Sections 3.2, 3.3 and professional report |

*Table 1: Group members contribution*

---

## 1. Analysis of the Problem

### 1.1. Growing Importance of Electricity Demand Forecasting

The ability to forecast the demand for electricity is an increasingly important problem for modern society, due to our increased reliance on electrical infrastructure across the industrial, commercial, and residential sectors. This increased reliance has been driven primarily by population growth and urban expansion, as well as the ever increasing availability and affordability of electrical tools and appliances to make people's lives and jobs more comfortable and easier. This higher than ever demand, coupled with a push for renewable energy, and increased pressure to minimise wasted power and emissions, makes being able to predict power usage trends accurately an incredibly important issue. In New South Wales (NSW), residential electricity usage accounts for approximately 24% of the total power usage across the state, and understanding how homes respond to weather events allows for improvement of energy planning, grid stability, and reduction of inefficiencies and unnecessary emissions through more coal being burned than the equivalent power required.

### 1.2. Project Objectives and Research Questions

This project aims to investigate the details of the relationship between weather patterns and power consumption in NSW, using datasets that are readily available to the general public. Our goal is to determine whether or not a measurable relationship exists between weather variables such as temperature, humidity, wind, and rainfall, and patterns of residential electricity usage. If there is a relationship found, the practical application of this would mean that based on weather predictions, power planning could be optimised to meet the needs of the state more closely, rather than overcompensating and causing waste, or underestimating and causing outages.

### 1.3. Why New South Wales Is a Relevant Case Study

This issue is particularly relevant in the state of NSW as it spans a large area, with many different climate conditions. NSW has a significant density of its population towards the eastern coast, and in summer periods, these areas can experience significant heat, causing power spikes and sometimes outages due to the widespread use of air conditioning. This coastal population density also makes the temperature for a majority of NSW homes quite volatile and susceptible to changes based on wind. Coastal cities are subject to rapid changes in air temperature brought on by offshore conditions. NSW however also spans to areas such as Barrington Tops and even parts of the Kosciusko mountain ranges, which see regular snowfall in the winter months, requiring significant power for heating devices. Spikes in energy usage caused by weather events can place significant strain on the power infrastructure across the state, and lead to an instability in supply. Being able to predict these fluctuations would provide an important benefit to both electricity providers and policymakers.

### 1.4. Infrastructure, Economic, and Environmental Impacts

Aside from concerns over the reliability of operation, significant fluctuations have a major environmental and economic impact. Electricity supply must always align as closely as possible with demand, as errors in forecasting can result in wildly inefficient energy production and drastically increased costs over time, through unnecessary power generation and wasted resources. Underestimating demand however, can drastically increase the risk of outages, and can force the reliance upon backup power generation systems for critical infrastructure such as hospitals and transport, which leads to independent fuel usage in urban areas, resulting in impacted air quality and an increase in overall carbon footprint. Improving demand prediction therefore contributes to a more efficient and sensible use of infrastructure, and a more climate conscious approach to power generation.

### 1.5. Renewable Energy and Weather Dependency

With an increased push to renewable energy, this problem becomes more relevant than ever. Many forms of renewable energy are inherently weather dependant, relying on sunshine and wind in order to generate power. Understanding the relationship between power usage and weather patterns, as well as the relationship between renewable energy generation and weather patterns, would allow for the optimisation of solar and wind power so they can be best utilised to reduce the NSW energy grid's contribution to global emissions and environmental harm.

### 1.6. Role of Publicly Available Datasets

The availability of public datasets is also an important factor when undertaking such a study. The project relies heavily on data that is available to the public in order to analyse trends and determine patterns. The data has already been collected by reputable organisations such as the Bureau of Meteorology (BOM) and various governmental and private energy stakeholders. These datasets contain enormous quantities of information, dating back to decades into the past, which allows the team to cut down on time spent collecting data, and analyse historical patterns with respect to energy usage over time.

### 1.7. Societal Relevance and Expected Outcomes

Overall, the investigation of weather effects on power consumption in NSW represents a real-life data engineering problem with very strong societal, economic and environmental relevance. By analysing the specifics of the relationship between weather and power usage, we will be able to draw helpful insights that could cut costs across the energy sector.

---

## 2. Description of the Hypotheses Being Tested and the Model Being Produced

### 2.1. Context

For this experiment we wish to understand and model the relationship between weather conditions and household energy demand. Currently in Australia Heating, Ventilation, and Air Conditioning (HVAC) accounts for over 40% of average home energy use (Milne and Reardon (2020)), as such accurately characterising the impact the use of this equipment has on the grid has significant relevance to designing strong and sustainable energy grids.

We theorise that on top of the existing 'duck curve' of grid load, where middle of the day grid load falls to near zero as rooftop and utility solar production peaks, temperature variations will result in a highly predictable change in demand. The impact of the duck curve on energy producers in Australia is also exacerbated by high solar adoption, with Australian solar production per capita being the highest in the world, over 1.3x the next highest country (Our World in Data (n.d.)).

We are also interested in other weather effects and their impact, including humidity, wind, and rain. Whilst we predict that temperature will have the most significant impact out of the listed weather effects, high humidity for example is expected to also produce a statistically significant impact on energy demand.

### 2.2. Hypotheses

This project tests a primary hypothesis concerning the relationship between temperature and residential electricity demand, along with a set of secondary hypotheses for the remaining weather variables and a structural hypothesis regarding climate zone differences.

#### 2.2.1 Primary Hypothesis: Temperature

The central hypothesis of this project is that temperature is the strongest weather-related driver of residential electricity consumption. This relationship is expected to be non-linear: demand is anticipated to be lowest at moderate, comfortable temperatures and to increase at both cold and hot extremes, as households activate heating and cooling appliances respectively. Formally:

- **H0:** Temperature has no statistically significant effect on residual household electricity consumption after controlling for time-of-day and seasonal patterns.
- **H1:** Temperature has a statistically significant effect on residual household electricity consumption, exhibiting a U-shaped relationship where consumption increases at both high and low temperature extremes.

#### 2.2.2 Secondary Hypotheses: Remaining Weather Variables

The project also tests whether humidity, wind speed, precipitation, and cloud cover each independently predict consumption beyond what temperature alone explains. For each variable x = relative humidity, wind speed, precipitation, cloud cover:

- **H0:** Variable x has no statistically significant effect on residual electricity consumption when all other weather variables are held constant.
- **H1:** Variable x has a statistically significant effect on residual electricity consumption when all other weather variables are held constant.

These secondary hypotheses are tested jointly through the regression models described below, with Lasso regression providing a principled method for determining which variables retain meaningful predictive power once weaker predictors are penalised.

#### 2.2.3 Structural Hypothesis: Climate Zone Comparison

Because the SGSC trial spans two distinct climate zones -- Zone 5 (Warm Temperate, Newcastle) and Zone 6 (Mild Temperate, Sydney) -- the project must first determine whether these zones exhibit sufficiently different demand behaviour to warrant separate modelling, or whether a single pooled model is appropriate:

- **H0:** Mean residual consumption does not differ significantly between Zone 5 and Zone 6 households.
- **H1:** Mean residual consumption differs significantly between Zone 5 and Zone 6 households.

This is tested using a two-sample t-test on residual consumption values prior to regression, with an alpha level of 0.05. The outcome determines whether subsequent modelling is conducted on pooled or zone-specific data.

### 2.3. Model Description

The project produces three complementary models, each operating on the residual consumption variable described in Section 3.3.

The first model is an **Ordinary Least Squares regression**, which regresses residual consumption against the weather predictors including a quadratic temperature term to capture potential non-linearity at temperature extremes. OLS was chosen as the starting point because its coefficients are directly interpretable, making it straightforward to assess the direction and approximate magnitude of each weather variable's relationship with consumption.

The second model applies **Lasso regression** to the same predictor set. Lasso adds a regularisation penalty that can shrink less informative coefficients to zero, providing a data-driven way to identify which weather variables retain predictive value once the full set is considered simultaneously. The regularisation parameter is selected through cross-validation. Lasso regression can also handle some multicollinearity whilst remaining interpretable (such as potentially correlated variables like cloud cover and relative humidity) (Bradley (2021)).

The third model is a **Random Forest**, an ensemble method that averages predictions across many decision trees built on bootstrapped subsamples. Unlike the regression models, Random Forest does not assume linear relationships, and its feature importance scores provide a complementary ranking of predictor relevance to compare against the OLS and Lasso results.

All three models are evaluated against test data withheld during the data preparation phase of the project to check for overfitting and will also be tested on data from a different Australian region to assess how well the findings might generalise beyond the SGSC trial sites.

---

## 3. Collection of the Data

### 3.1. Collecting Data via Public Datasets

This project relies on publicly available datasets to explore the relationship between weather conditions and the residential electricity consumption in NSW. The use of existing datasets allows our team to access large quantities of historical data collected through a reputable organisation without the time, cost, and ethical problems associated with conducting an independent large-scale household energy study. The datasets chosen for this project include Smart Grid Smart City (SGSC) Customer Trial data and Open-Meteo Historical Weather API weather data: Australian Government (2015) and Open-Meteo (2026).

The SGSC dataset was collected as part of an Australian Government initiative between 2010 and 2014 and contains 30-minute electricity consumption readings from more than 6,000 households across NSW (Australian Government, 2014). In addition to electricity usage data, the dataset includes metadata relating to climate zones, rooftop solar participation, and whether households were part of control or treatment groups during the trial. This additional information is critical because it allows our team to filter households and remove confounding influences within analysis. Due to the high number of observations, the dataset is also usable for examining longer-term relationships between energy consumption and weather variables.

The Open-Meteo Historical Weather API will be responsible for collecting any historical weather data. According to Open-Meteo (2025), the service provides hourly weather observations for specific geographic coordinates. The API was chosen because of its accessibility, hourly temporal resolution, and is capable of obtaining several weather variables simultaneously. The variables extracted for this project will include temperature, relative humidity, cloud cover, wind speed, and precipitation, as these are the factors most likely to influence residential electricity demand through heating, cooling, lighting, and general household behaviour. The structured nature of Open-Meteo API was preferred over alternative sources such as the Bureau of Meteorology (BOM), due to ease of integration and compatibility with the SGSC timestamp format. Weather data will be retrieved for representative locations corresponding to the warm temperate and mild temperate climate zones identified within the SGSC dataset.

To improve the reliability of the analysis, the dataset will first be filtered to exclude households with rooftop solar systems and households involved in tariff intervention programs. The remaining households will be grouped according to climatic zone and aggregated into average values of electricity consumption over time. Since the SGSC dataset is taken at 30-minute intervals and the weather data is taken hourly, the electricity consumption data will be aggregated into hourly intervals to obtain consistency between both datasets before analysis is conducted.

The quantity of data available is sufficient for meaningful statistical analysis and modelling. The SGSC dataset spans approximately four years of continuous 30-minute measurements across thousands of households, resulting in millions of individual observations prior to aggregation. The large sample size reduces the impact of deviant household behaviour and relationships can be explored across multiple seasons and weather events. Likewise, the weather dataset provides continuous hourly observations over the same period, ensuring strong temporal alignment between weather events and electricity demand.

Although the datasets are historical rather than contemporary, they remain appropriate for analysing the broader relationship between weather conditions and residential electricity consumption. Our team accepts that electricity demand today may be different from 2014 given how modern household technologies and energy usage patterns have changed since then. However, these changes do not invalidate the usefulness of the dataset for investigating weather-driven energy behaviour.

### 3.2. Dependent and Independent Variables in the Study

Now that we have our data, we can focus on understanding it. The first goal is to identify the dependent variable (the outcome of interest) and the independent variables. Independent variables are the factors on whose we want to quantify the effect on the outcome variable.

In our study, the variable of interest is the energy consumption and the independent variables are weather-related variables. Indeed, we're trying to quantify the impact of the weather on the energy consumption. However, we have to make several choices to select our variables:

- **Aggregated vs. household-level consumption.** The dataset provides readings at the individual household level, but modeling at that granularity introduces a lot of noise from factors like occupancy or appliance ownership that are hard to control for. We therefore choose to aggregate consumption across households sharing the same climate zone, which makes the weather signal much easier to isolate. To identify the climate zone, we can use the `ASSRTD_CLIMATE_ZONE_CD` variable in the sgsc customer household dataset that separates the households between two climate groups: warm temperate and mild temperate. Once our data is separated, we can just compute the mean consumption of each group over time. This leads to having for each time slot a mean value for each group representing the mean energy consumption in the climate zone. However, it raises a problem about the weather data. Indeed, we can't get the location of every household for obvious ethics reasons, but we still need relevant weather data on the area we're studying. With the actual data structure, we need weather data from two locations (one for the warm temperate group and one for the mild temperate group). According to the SGSC trial data, the warm temperate zone covers the coastal strip of New South Wales encompassing Wollongong, Sydney, and Newcastle up to 32 degrees south and the mild temperate zone covers the coastal strip of southern and hinterland New South Wales west of Sydney as far north as 28 degrees south. Table 2 shows the two locations from where we're going to extract the weather data. Doing this will enhance our data analysis because we'll have accurate weather data to analyze its impact.

- **Which weather variables to include.** Temperature is the obvious main driver since it directly governs heating and cooling demand. We also include relative humidity, as it affects perceived temperature and air conditioning usage, and cloud cover, which influences both lighting behavior and solar heat gain. We also include wind speed and precipitation to test their effect on the energy consumption. Others like atmospheric pressure or visibility are excluded since there is no clear causal link to residential electricity use.

- **Net vs. gross consumption.** Some households in the dataset have rooftop solar panels, which creates a distinction between gross consumption and net consumption after subtracting on-site generation. Since solar generation is itself weather-dependent, using net consumption would introduce a confounding relationship between our variables. We restrict our analysis to non-solar households, using the `GENERAL_SUPPLY_KWH` reading which reflects grid consumption only.

| Climate Zone | Location | Coordinates |
|---|---|---|
| Zone 5 - Warm Temperate | Newcastle | -32.9283, 151.7817 |
| Zone 6 - Mild Temperate | Sydney CBD | -33.8688, 151.2093 |

*Table 2: Locations chosen*

Finally, our **dependent variable** is aggregated household energy consumption, and our **independent variables** are temperature, relative humidity, cloud cover, wind speed and precipitation.

### 3.3. Confounding Variables and How to Handle Them

In order to understand the impact of the weather on the household energy consumption, we need to isolate its effects. To do so, we need to consider the confounding variables that could confound the results. We can split them in two groups:

- **Control variables:** Beyond weather, energy consumption is heavily influenced by temporal patterns that we need to account for before drawing any conclusions. The most significant ones are the hour of the day, which captures the day/night cycle and typical daily routines such as morning and evening peaks, the day of the week, since weekday and weekend consumption profiles differ substantially, and the month of the year, which reflects seasonal behavioral shifts independently of temperature. Public holidays also behave similarly to weekends and should be treated as such. To control for these patterns, we compute a stratified baseline consumption for each combination of hour, day type (weekday vs. weekend/holiday) and season. For each observation, we then subtract the corresponding baseline to obtain a residual consumption that represents the deviation from the expected routine pattern. This normalized variable is what we will use as our dependent variable when modeling the effect of weather, ensuring that the relationships we identify reflect genuine weather-driven behavior rather than artifacts of daily or seasonal routines.

- **Confounding variable:** An additional confounding factor is specific to the SGSC trial design. Participants were divided into treatment groups exposed to cost-reflective tariffs and demand response incentives, meaning a portion of households deliberately modified their consumption in ways that have nothing to do with weather. Keeping these households in the analysis would introduce noise that could bias our results. We therefore restrict our sample to the control group, identified by the `CONTROL_GROUP_FLAG` column in the household dataset, which ensures that the consumption patterns we observe reflect natural behavior rather than responses to pricing interventions.

By addressing both categories of confounding variables, we ensure that the variation remaining in our dependent variable is as cleanly attributable to weather as possible. The stratified baseline normalization removes the structural noise introduced by routine temporal patterns, while restricting the sample to the control group eliminates behavioral distortions induced by the trial design. This puts us in a solid position to move on to the analysis itself.

---

## 4. Techniques Selected

This project uses a pipeline analysis from exploratory to predictive modelling. The techniques are all chosen for their intended use and contribute together to developing the coherent workflow of the data preparation stages outlined in the preceding sections.

### 4.1. Exploratory Analysis

The first phase is exploratory analysis of Pearson correlation coefficients and PCA, or principal component analysis of the data. Correlation analysis enables us to observe whether there are pairwise linear relations between each weather variable and the residual consumption variable, which is the deviation from the stratified baseline defined in Section 3.3. This provides preliminary insight into which predictors may be important, and if any predictors are closely colinear with one another. In our regression models, high collinearity among predictors can lead to distortion of coefficient estimates; catching it early can help us make better decisions about the variable inclusion. PCA supports this, but can also segment the weather variables into specific, but similar, patterns and predictors so that we have a greater understanding of how the information is structured, which may help us detect whether specific features are effectively reflecting the same variability. The reason is that this is especially helpful for a predictor set where temperature, humidity and cloud cover are correlated and PCA tells us if the data are well-represented by a smaller number of dimensions. However, we don't use the principal components as part of regression models so the outcome of this step informs our interpretations of the behaviour of the predictors for the rest of the analysis.

### 4.2. Hypothesis Tests

We test the hypotheses before transitioning to the regression analysis and validate one of the major structural assumptions of our design to be true. In particular, we use a two-sample t-test to determine whether mean residual consumption for the two climate groups (Zone 5 warm temperate and Zone 6 mild temperate) differs significantly. This is relevant since, if we aggregate over climate zones without confirming this difference, it could obscure meaningful differences, or alternatively force us to separate them altogether if that seems unnecessary. The test is fitting here, with residual consumption, when considered after the baseline normalisation, being able to be considered approximately normally distributed, and since we are comparing two independent groups. If groups do differ significantly, it is defensible to treat them separately during the modelling stage. We use an alpha level of 0.05 as our significance threshold throughout.

### 4.3. Regression Modeling

For regression modeling we start with Ordinary Least Squares (OLS). OLS is an intelligent place to start because it is easily interpretable. The estimated coefficients provide us a simple, objective measure of the amount -- one unit change in one of the weather factors that corresponds to an amount change in the residual consumption, assuming all other variables stand constant. This interpretability is vital on a study like ours where consumption is not only determined, but also the fact that what weather conditions are effective are what we should expect them to be in practice. For instance, a strongly positive coefficient for temperature would indicate if the demand of cooling electricity increases on hotter days. We will also perform OLS testing to check for standard regression diagnostics, residual plots, and variance inflation factors, checking whether our assumptions are in accordance with reality and if any predictors must be transformed or removed before our model is finalised.

We adopt Lasso regression (Least Absolute Shrinkage and Selection Operator) as a variable-selection method after OLS. Lasso imposes a regularisation penalty, forcing the coefficients to zero for non-significant predictors; in some particular situations, they become perfectly zero, removing them from the model. This is handy in our case we have many weather predictors which may not all perform when the other things are taken into account. Wind speed and precipitation, for example, were introduced as exploratory variables in Section 3.2 because their effect on consumption is uncertain, and Lasso offers a principled and empirical manner to determine whether they deserve to be part of the final model. Lasso assists us in coming to a more parsimonious model minimizing overfitting, enabling the final model to be easier to be interpreted and to be deployed. Lambda, another regularisation parameter, will be used with cross-validation to make sure that the penalty is suitable for our data.

### 4.4. Random Forest Model

We finalise a Random Forest model fitting to evaluate its predictive performance and feature importance rankings. Random Forests are an ensemble approach that constructs various decision trees based on bootstrapped subsets of the data and averages their predictions. This takes care of nonlinear relationships and interactions between variables, without explicitly prescribing any information, and so, is a nice complement to the linear models above. Random Forests are less interpretable than OLS in terms of individual coefficients, but they offer feature importance scores based on permutation that indicate how much the individual predictors contribute to model predictive accuracy overall. These ranked importance rankings compared to the OLS coefficients and the Lasso selection gives us a much more nuanced understanding about which weather variables are driving household energy consumption in NSW.

---

## 5. Discussion on the Ethics of the Problem

Data engineering brings upon a strong dilemma of ethics, as personal data, if not used responsibly or respectfully, can be used for dishonest gains, to harm or treat individuals unfairly. This project joins household electricity consumption records with historical weather data. Whilst both datasets are publicly available, their combination can focus on location and private usage. The 5 C's -- Consent, Clarity, Consistency, Control and Consequences -- provide a clear structure when addressing these concerns (Loukides et al. (2018)).

### 5.1. Consent

The Smart Grid Smart City dataset was collected between 2010 and 2014 as part of a funded trial by Ausgrid. Whilst participants agreed to have their electricity data used within that trial, they may not have consented to their data being used beyond that, such as university research projects a decade later. Although the dataset is published under a Creative Commons license that permits secondary use, legality is not the same as ethical consent. Since re-contacting participants is not feasible, the team should acknowledge this limitation and avoid expanding the analysis beyond its stated purpose, as advised by American Statistical Association (2018).

### 5.2. Clarity

Clarity requires that the participants understand what they consented to. To address the consent gap, the team plans to produce a data statement that clearly describes the data's origin, the purpose of its collection, current research goal, and the limitations. This ensures anyone reviewing the project understands what the data is, avoiding misrepresentations of data. To demonstrate active participant care and clarity, the team will document its data-handling protocols transparently so that supervisors and/or participants could verify or comment on responsible practices, as recommended by UK Data Service (2021).

### 5.3. Consistency

The data should be consistently used to ensure the project maintains its purpose. The project should only do what it says it will do. Using the data to draw external conclusions about household patterns, financial status, or lifestyle would be inconsistent with the research question and would be ethically questioned, due to undermining the trust of participants and open data publications. The weather data must also be joined to the electricity data consistently with matching postcodes and aligned timestamps, as results would be misleading if otherwise. American Statistical Association (2018) guidelines require us to be upfront if such an issue arises.

### 5.4. Control

Being public data, the participants of the SGSC dataset have no control on what is happening with their data, being unable to do something about it if they change their mind. O'Neil (2016) argues that data-driven systems where decisions are made by models no one can inspect or challenge are the most harmful. As participants are not able to withdraw their records and have no direct control over how it is used or displayed, it is the team's responsibility to be transparent with the data, using interpretable methods with clearly reported coefficients, so that any findings can be understood or questioned. The team should also only retain the data fields actually needed for the analysis, avoiding storing or sharing records beyond what the project requires.

### 5.5. Consequences

The team should carefully assess what could go wrong when utilizing the data and what is presented, even if it is not immediately obvious. Loukides et al. (2018) warns that combining two datasets often creates risks far greater than on their own. Half-hourly electricity data combined with postcodes can reveal when a household is occupied, what appliances are running, and when people are asleep or away from home. If misused, this research could enable harm, such as burglary, stalking, or harassment. Whilst the data already exists, the team should be careful not to make inferences that would highlight individual household patterns. On a larger scale, if weather-sensitive models are developed and adopted commercially by energy retailers, they could support dynamic pricing schemes that charge more to households depending on upcoming weather patterns that increase demand. Whilst this is economically rational and viable from a business perspective, the long-term depends on balancing profits with negative consumer feedback. In reality, this would be the most affective for consumers with the least ability to respond, such as low-income households or households dependent on powered medical equipment. Their consumption does not drop when prices rise due to necessity. O'Neil (2016) describes this as a defining feature of harmful data-driven systems, that extract data efficiency from those with no room to adapt. For this reason, the American Statistical Association (2018) guidelines advise researchers to consider the effects of their analysis and to avoid outputs that could be used to negatively impact vulnerable populations. Finally, presenting findings that could contribute to the labelling of specific suburbs as a high-consumption area, or energy inefficient, without accounting for variable weather, high-density housing or socioeconomic factors, would be misleading and potentially harmful. The team will avoid publishing individual household results or demographics and avoid broad claims about NSW patterns, whilst acknowledging the sample's skew due to being in trial sites such as Newcastle.

---

## Conclusion

This experimental design report has outlined a structured approach to investigate the relationship between weather and residential electricity consumption in New South Wales using publicly available datasets. By combining Australian Government (2015) household electricity data with historical weather observations, this project aims to determine the extent to which variables such as temperature, humidity, wind speed, cloud cover, and precipitation influences household energy demand.

The methodology incorporates data filtering, aggregation, baseline normalisation, and several modeling techniques to ensure that weather-driven behaviour can be isolated from everyday energy consumption patterns. Statistical and machine learning techniques, which include OLS regression, Lasso regression, and Random Forest modeling, were selected to provide both interpretable and predictive insights into energy usage behaviour.

Overall, this project represents a practical and socially significant data engineering problem with strong relevance to energy forecasting, infrastructure efficiency, and environmental sustainability. By improving the understanding of how weather conditions influence residential electricity demand, the project has the potential to support more informed decision-making within the NSW energy sector.

---

## References

American Statistical Association. (2018). *Ethical guidelines for statistical practice.* https://www.amstat.org/your-career/ethical-guidelines-for-statistical-practice

Australian Government. (2015). *Smart grid, smart city customer trial data.* https://www.data.gov.au/data/dataset/smart-grid-smart-city-customer-trial-data

Bradley, B. (2021, December). When to use lasso regression. *Towards Data Science.* https://towardsdatascience.com/when-to-use-lasso-regression-8a4c4b334fa8/

Loukides, M., Mason, H., & Patil, D. J. (2018). *Ethics and data science.* O'Reilly Media.

Milne, G., & Reardon, C. (2020). Heating and cooling. *YourHome.* https://www.yourhome.gov.au/energy/heating-and-cooling

O'Neil, C. (2016). *Weapons of math destruction.* Crown Books.

Open-Meteo. (2026). *Historical weather API.* https://open-meteo.com/en/docs/historical-weather-api

Our World in Data. (n.d.). *Per capita electricity generation from solar.* https://ourworldindata.org/grapher/solar-electricity-per-capita

UK Data Service. (2021). *Research ethics and integrity case studies.* https://ukdataservice.ac.uk/
