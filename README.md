# 🏥 **UbuntuCare Smarter Wait Times Intelligence**  
**🔍 Data-Driven Insights for Efficient & Equitable Healthcare Delivery**  
*📍 South Africa | 👥 50,000+ Synthetic Patient Records | 📅 Modeled 2025*  

---

## 📌 **Overview**  
**UbuntuCare Smarter Wait Times** is a **machine learning and healthcare operations intelligence project** designed to **analyze, predict, and reduce patient wait times** using realistic **synthetic data**.

By combining **XGBoost predictive modeling**, **patient-flow analytics**, and **actionable operational insights**, this project supports healthcare leaders in:  
- **Reducing excessive wait times** ⏳  
- **Improving fairness and patient experience** ❤️  
- **Optimizing staffing and resource allocation** 🏥  
- **Strengthening system-wide efficiency** ⚙️  

> **Note:** Due to GitHub file-size limits, the full synthetic dataset is hosted on **Hugging Face**.

---

## 🚀 **Key Features**  
✅ **Wait Time Prediction Model** using **XGBoost** (R² ≈ **0.62**, MAE ≈ **134 mins**)  
✅ **Feature Importance** from clinical, demographic & operational factors  
✅ **Facility-level performance diagnostics**  
✅ **Equity analysis** on vulnerable groups (age, chronic conditions, rural access)  
✅ **Operational risk insights** for weekend, seasonal & peak-hour surges  

---

## 📊 **Dataset Summary**  
- **50,000+ synthetic patient encounters**  
- Includes **demographics, vitals, triage status, ICD-10 codes, facility attributes & timestamps**  
- Modeled to reflect **realistic patterns in South African public healthcare**  
- **Stored on Hugging Face** due to GitHub space limits  
- **Target Variable:** `wait_time_minutes`  

---

## 💼 **Key Insights & Business Impact**  

📌 **Demographics Matter**  
- Older patients (40+) and males wait **10–36 mins longer**  
- Rural facilities show **+8 mins** delay  

📌 **Clinical Drivers**  
- Chronic patients (diabetes, hypertension) face waits **>500 mins**  
- High SATS = fast priority; low SATS = long queues  

📌 **Operational Patterns**  
- Weekends: **+20% slower**  
- Peak hours (10–11 AM): **>330 mins**  
- Winter flows better despite higher volume  

📌 **Financial Impact (Modeled)**  
- **R54.1 million net benefit** from optimized operations  

---

## 🔮 **Recommendations**  

### 🏥 **Facility & Patient Targeting**  
- Focus on high-delay facilities  
- Prioritize older & chronic patients  
- Reduce rural–urban gaps in service efficiency  

### ⚙️ **Operational Improvements**  
- Smarter staffing for peak hours / weekends  
- Apply lessons from high-efficiency periods (winter, overnight)  
- Monitor facility outliers  

### 🩺 **Clinical Flow Optimization**  
- Dedicated pathways for chronic-condition patients  
- Preserve triage priority for high-risk cases  

### 🤖 **Predictive Modeling & Decision Support**  
- Use XGBoost for **screening & scheduling** support  
- Human-in-the-loop for clinical decisions  
- Ethical handling of demographic variables  

---

## 🛠 **Tech Stack**  
- **Python 3.10**  
- **Pandas, NumPy, Matplotlib, Seaborn**  
- **XGBoost**  
- **Scikit-Learn**  
- **Power BI / Plotly** (optional dashboards)  

---

✍ **Author:** Lesiba James Kganyago  
📅 **Year:** 2025  
