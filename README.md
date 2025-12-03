# Spatio-Temporal-Prediction-of-Land-Subsidence-using-Kriging-and-LSTM
**Part of my Master's program application, Oct 2024 @ NCKU Resource Engineering**
![Made with R](https://img.shields.io/badge/Made%20with-R-276DC3?logo=r&logoColor=white)

**Note:The data is for demonstration purposes, not accurate.**


This project develops a spatio-temporal prediction framework for land subsidence by integrating **geostatistical interpolation** and **deep learning**. The study focuses on the Yunlin region in Taiwan, where groundwater over-extraction has caused severe subsidence.



The model combines:
- **Ordinary Kriging (OK)** for spatial interpolation  
- **Long Short-Term Memory (LSTM)** networks for temporal forecasting  
- **Hydro-meteorological variables** including groundwater levels, rainfall, and pumping rates  

---

## Research Background

Groundwater over-pumping in Taiwan has resulted in significant land subsidence, especially in the **Choshui River alluvial fan**.  
Traditional spatial-only analysis often overlooks **temporal trends** and regions lacking monitoring wells.

This project:
- Builds a **spatio-temporal prediction model**
- Incorporates **rainfall**, **pumping**, and **groundwater level** as features
- Aims to improve prediction reliability and provide insights into subsidence mechanisms

---

## Methodology

### Ordinary Kriging (OK)

Ordinary Kriging assumes:
- Unknown but constant mean
- Spatial correlation depends only on distance
- Provides **Best Linear Unbiased Estimator (BLUE)**

The OK system:

<div align="center">
  
<img width="1040" height="194" alt="圖片" src="https://github.com/user-attachments/assets/5193f51a-5d7d-4d1a-bd87-ecbda9d632ae" />

</div>

This is used to interpolate **rainfall** and **pumping rate** at groundwater well locations.
<img width="1742" height="600" alt="圖片" src="https://github.com/user-attachments/assets/d95115bc-613b-4c16-a643-f4d4f48c2c22" />

---

### Long Short-Term Memory (LSTM)

LSTM networks capture long-term dependencies via:
- Forget gate  
- Input gate  
- Output gate  

A hidden state evolves as:

$$f_t = σ(W_f · [h_{t-1}, x_t] + b_f)$$
$$i_t = σ(W_i · [h_{t-1}, x_t] + b_i)$$
$$Ĉ_t = tanh(W_C · [h_{t-1}, x_t] + b_C)$$
$$C_t = f_t * C_{t-1} + i_t * Ĉ_t$$
$$o_t = σ(W_o · [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t * tanh(C_t)$$

The model is trained using **2016–2019** data and predicts **2020 groundwater levels**.

---

## Data Description

**Study period:** 2016–2020  
**Region:** Yunlin, Taiwan  

Data sources:
- Groundwater level – Taiwan Water Resources Agency  
- Rainfall – WRA hydrological database  
- Pumping rates – provided by Dr. Lin (NCKU Hydraulic Lab)

Data included:
- 10 groundwater monitoring wells  
- Rain gauges  
- Pumping wells near target stations  

---

## Model Training

LSTM hyperparameters were tuned individually for each well.  
Example training results (insert your images here):

---

## Evaluation Metrics

The model is evaluated using:

### **Mean Squared Error (MSE)**

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### **Root Mean Squared Error (RMSE)**

$$\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }$$

### **Coefficient of Determination (R^2)**

$$R^2 = 1 -\frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

---

## Prediction & Feature Contribution

LSTM predicts **2020 groundwater levels** and uses sensitivity analysis to estimate each feature’s contribution.
<img width="1981" height="532" alt="圖片" src="https://github.com/user-attachments/assets/f293b581-2a4e-4664-88a5-e96bdb26f305" />

Example table (modify with your actual numbers):

<div align="center">
  
| Well | RMSE | Groundwater | Rainfall | Pumping | Dominant Feature |
|------|------|-------------|----------|---------|------------------|
| Fang-Cao | 0.0059 | 0.17 | 0.09 | 0.10 | Groundwater |
| Tuku | 0.0839 | 0.067 | 0.056 | 0.182 | Pumping |
| Yuan-Chang | 0.007 | 0.021 | 0.062 | 0.091 | Pumping |

</div>
<img width="1871" height="407" alt="圖片" src="https://github.com/user-attachments/assets/81fa440e-a9c4-446c-aba0-dd466b9fb9ff" />


## Future Work

### **Data Improvements**
- Consider multiple aquifers in the Choshui River alluvial fan  
- Include all pumping categories: irrigation, aquaculture, industrial, domestic  
- Add CWB rainfall data for completeness  
- Integrate water balance equation:

$$\Delta S(t) = P(t) - R(t) - E(t) - Q(t)$$

---

### **Methodology Enhancements**
- Try **Co-Kriging** when multiple correlated variables exist  
- Add **physical constraints** from soil mechanics & hydrogeology  
- Explore more sensitive feature-importance methods  

---

## Workflow Summary

Data Collection → Preprocessing → Kriging Interpolation→ LSTM Modeling → Model Validation → Feature Contribution→ Land Subsidence Formula Development
