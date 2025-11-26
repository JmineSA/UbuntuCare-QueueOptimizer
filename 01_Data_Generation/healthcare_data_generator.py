#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced South African Public Healthcare Dataset Generator
Synthetic dataset of ~250,000 patient visits simulating the overburdened 
South African public healthcare system with realistic patterns and systemic flaws.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import re
from faker import Faker
import sqlite3
from sqlite3 import Error
import pymongo
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Faker for realistic names
fake = Faker()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_RECORDS = 250000

# Province and facility mappings
PROVINCES = {
    'Gauteng': 0.35,
    'KwaZulu-Natal': 0.25,
    'Western Cape': 0.15,
    'Eastern Cape': 0.12,
    'Limpopo': 0.13
}

FACILITIES = {
    'Gauteng': [
        'Chris Hani Baragwanath ACAD (Soweto)', 'Steve Biko Academic Hosp', 
        'Helen Joseph Hospital', 'Natalspruit Hospital', 'Mamelodi Day Clinic',
        'Diepkloof Clinic', 'Kalafong Hospital', 'Charlotte Maxeke Hospital',
        'Tembisa Hospital', 'Sebokeng Hospital'
    ],
    'KwaZulu-Natal': [
        'King Edward VIII Hosp (DBN)', 'Inkosi Albert Luthuli Central Hospital',
        'Addington Hospital', 'R K Khan Hospital', 'Umlazi Mega Clinic',
        'Prince Mshiyeni Memorial Hospital', 'Greys Hospital'
    ],
    'Western Cape': [
        'Groote Schuur Hospital (CPT)', 'Tygereberg Hospitl', 
        'Khayelitsha Site B CHC', 'Gugulethu CHC', 'Red Cross Childrens Hospital',
        'Mitchells Plain Hospital', 'Victoria Hospital'
    ],
    'Eastern Cape': [
        'Livingstone Hospital (Gqeberha)', 'Frere Hospital', 
        'Dora Nginza Hospital', 'Nelson Mandela Academic Hospital',
        'Uitenhage Provincial Hospital'
    ],
    'Limpopo': [
        'Polokwane Provincial Hospital', 'Mankweng Hospital',
        'Seshego Hospital', 'Tshilidzini Hospital', 'Letaba Hospital'
    ]
}

# Facility typos and variations
FACILITY_TYPOS = {
    'Chris Hani Baragwanath ACAD (Soweto)': ['Chris Hani Bara', 'Baragwanath Hospital', 'CHB Hospital'],
    'Steve Biko Academic Hosp': ['Steve Biko Hosp', 'S Biko Academic'],
    'Groote Schuur Hospital (CPT)': ['Groote Schuur', 'GSH CPT'],
    'Tygereberg Hospitl': ['Tygerberg Hospital', 'Tygereberg', 'Tygerberg Hosp'],
    'King Edward VIII Hosp (DBN)': ['King Edward Hosp', 'KEH DBN'],
    'Mamelodi Day Clinic': ['Mamelodi Clininc', 'Mamelodi Clinic'],
    'Khayelitsha Site B CHC': ['Khayelitsha CHC', 'Site B Clinic'],
    'Diepkloof Clinic': ['Diepkloof Clininc'],
    'Umlazi Mega Clinic': ['Umlazi Clinic', 'Umlazi Mega'],
    'Gugulethu CHC': ['Gugulethu Clinic', 'Gugs CHC']
}

# Departments with variations
DEPARTMENTS = {
    'Emergency': ['ER', 'Emergncy', 'Emergency Dept', 'A&E'],
    'Trauma': ['Trauma Unit', 'Trauma Centre'],
    'Surgery': ['Surgical', 'Surgical Ward'],
    'Paediatrics': ['Pediatrics', 'Paeds', 'Child Health'],
    'Antenatal': ['Maternity', 'Obstetrics', 'ANC'],
    'OPD': ['Outpatient', 'Outpatient Dept'],
    'HIV/TB Clinic': ['TB Clinic', 'HIV Clinic', 'ARV Clinic'],
    'Pharmacy': ['Dispensary', 'Medication Collection']
}

# ICD-10 codes and complaints mapping
COMPLAINT_ICD_MAPPING = {
    'Cough & fever': ['A15.0', 'J18.9', 'R05', 'R50.9'],
    'stab wound chest': ['S21.90', 'S21.91', 'T14.8'],
    'MVA passenger': ['V43.52', 'V43.62', 'V89.2'],
    'headache': ['R51', 'G44.209'],
    'shortness of breath': ['R06.02', 'J98.9'],
    'routine antenatal': ['Z34.90', 'O09.90'],
    'diabetes checkup': ['E11.9', 'Z13.1'],
    'TB meds collection': ['Z76.0', 'Z79.899'],
    'severe abd pain': ['R10.0', 'R10.9', 'K35.90'],
    'child not feeding': ['R63.3', 'P92.9']
}

# Additional common complaints
ADDITIONAL_COMPLAINTS = [
    'hypertension review', 'asthma attack', 'diarrhoea', 'vomiting', 'rash',
    'back pain', 'chest pain', 'dizziness', 'eye problem', 'dental pain',
    'fractured arm', 'burns', 'seizure', 'UTI symptoms', 'STI screening',
    'mental health crisis', 'allergic reaction', 'animal bite'
]

# Enhanced clinical severity mapping
CLINICAL_SEVERITY = {
    'Red': {'min': 1, 'max': 3, 'resp_rate': (30, 40), 'heart_rate': (140, 180), 'sats': (70, 89)},
    'Orange': {'min': 4, 'max': 6, 'resp_rate': (25, 30), 'heart_rate': (120, 140), 'sats': (90, 94)},
    'Yellow': {'min': 7, 'max': 9, 'resp_rate': (20, 25), 'heart_rate': (100, 120), 'sats': (95, 97)},
    'Green': {'min': 10, 'max': 12, 'resp_rate': (12, 20), 'heart_rate': (60, 100), 'sats': (98, 100)},
    'Blue': {'min': 10, 'max': 12, 'resp_rate': (12, 20), 'heart_rate': (60, 100), 'sats': (98, 100)}
}

# Province coordinates for geospatial data
PROVINCE_COORDS = {
    'Gauteng': {'lat': -26.2041, 'lon': 28.0473, 'variation': 1.5},
    'KwaZulu-Natal': {'lat': -28.5306, 'lon': 30.8958, 'variation': 2.0},
    'Western Cape': {'lat': -33.9249, 'lon': 18.4241, 'variation': 1.8},
    'Eastern Cape': {'lat': -32.2968, 'lon': 26.4194, 'variation': 2.2},
    'Limpopo': {'lat': -23.4013, 'lon': 29.4179, 'variation': 2.5}
}

# Facility capacity levels
FACILITY_CAPACITY = {
    'Chris Hani Baragwanath ACAD (Soweto)': 'High',
    'Steve Biko Academic Hosp': 'Medium',
    'Helen Joseph Hospital': 'Medium',
    'Natalspruit Hospital': 'Low',
    'Mamelodi Day Clinic': 'Very Low',
    'Diepkloof Clinic': 'Very Low',
    'Kalafong Hospital': 'Medium',
    'Charlotte Maxeke Hospital': 'High',
    'Tembisa Hospital': 'Medium',
    'Sebokeng Hospital': 'Low',
    'King Edward VIII Hosp (DBN)': 'High',
    'Inkosi Albert Luthuli Central Hospital': 'High',
    'Addington Hospital': 'Medium',
    'R K Khan Hospital': 'Medium',
    'Umlazi Mega Clinic': 'Low',
    'Prince Mshiyeni Memorial Hospital': 'Medium',
    'Greys Hospital': 'Medium',
    'Groote Schuur Hospital (CPT)': 'High',
    'Tygereberg Hospitl': 'High',
    'Khayelitsha Site B CHC': 'Low',
    'Gugulethu CHC': 'Low',
    'Red Cross Childrens Hospital': 'Medium',
    'Mitchells Plain Hospital': 'Medium',
    'Victoria Hospital': 'Low',
    'Livingstone Hospital (Gqeberha)': 'Medium',
    'Frere Hospital': 'Medium',
    'Dora Nginza Hospital': 'Low',
    'Nelson Mandela Academic Hospital': 'High',
    'Uitenhage Provincial Hospital': 'Low',
    'Polokwane Provincial Hospital': 'Medium',
    'Mankweng Hospital': 'Medium',
    'Seshego Hospital': 'Low',
    'Tshilidzini Hospital': 'Low',
    'Letaba Hospital': 'Low'
}

def introduce_typos(text, typo_prob=0.1):
    """Introduce typos into text with given probability"""
    if random.random() > typo_prob or not text:
        return text
    
    typo_operations = [
        lambda s: s[:-1] if len(s) > 2 else s,  # drop last character
        lambda s: s + s[-1] if len(s) > 2 else s,  # duplicate last character
        lambda s: s.replace('a', 'e').replace('e', 'a') if len(s) > 3 else s,  # swap vowels
        lambda s: s.upper() if random.random() < 0.3 else s,  # random capitalization
        lambda s: s.lower() if random.random() < 0.3 else s
    ]
    
    return random.choice(typo_operations)(text)

def generate_patient_id():
    """Generate patient ID with some duplicates"""
    ids = [f"SAH-{random.randint(1000000, 9999999)}" for _ in range(int(NUM_RECORDS * 0.95))]
    
    # Add duplicates
    duplicate_ids = random.sample(ids, int(NUM_RECORDS * 0.05))
    all_ids = ids + duplicate_ids
    random.shuffle(all_ids)
    
    return all_ids[:NUM_RECORDS]

def generate_province_facility():
    """Generate province and facility with realistic distribution"""
    province = random.choices(list(PROVINCES.keys()), weights=list(PROVINCES.values()))[0]
    
    facilities = FACILITIES[province]
    facility = random.choice(facilities)
    
    # Introduce typos and variations
    if facility in FACILITY_TYPOS and random.random() < 0.3:
        facility = random.choice(FACILITY_TYPOS[facility])
    
    return province, introduce_typos(facility)

def generate_age():
    """Generate age with bimodal distribution and errors"""
    if random.random() < 0.01:  # 1% errors
        return random.choice([-1, 999, 205])
    if random.random() < 0.02:  # 2% missing
        return np.nan
    
    # Bimodal distribution: peak 0-5 and 20-35
    if random.random() < 0.4:
        return random.randint(0, 5)  # Pediatric peak
    elif random.random() < 0.7:
        return random.randint(20, 35)  # Young adult peak
    else:
        return random.randint(36, 90)  # Older ages

def generate_gender():
    """Generate gender with variations and errors"""
    options = [
        ('M', 0.35), ('Male', 0.15), ('F', 0.35), ('Female', 0.13),
        ('fem', 0.01), ('MALE', 0.005), ('U', 0.01), ('', 0.005),
        ('Intersex', 0.002), ('Other', 0.002)
    ]
    
    choices, weights = zip(*options)
    return random.choices(choices, weights=weights)[0]

def generate_race():
    """Generate race with weighted distribution"""
    races = ['Black', 'Coloured', 'White', 'Indian', 'Unknown']
    weights = [0.75, 0.15, 0.05, 0.04, 0.01]
    return random.choices(races, weights=weights)[0]

def generate_complaint_and_icd(age, gender, race, season):
    """Generate chief complaint and ICD-10 code with realistic patterns"""
    # Seasonal patterns
    if season == 'Winter' and random.random() < 0.6:
        complaint = 'Cough & fever'
    elif age <= 5 and random.random() < 0.4:
        complaint = random.choice(['Cough & fever', 'child not feeding', 'diarrhoea', 'vomiting'])
    elif 18 <= age <= 35 and gender in ['F', 'Female', 'fem'] and random.random() < 0.3:
        complaint = 'routine antenatal'
    elif age > 40 and random.random() < 0.3:
        complaint = random.choice(['hypertension review', 'diabetes checkup', 'chest pain'])
    elif random.random() < 0.1:  # HIV/TB burden
        complaint = 'TB meds collection'
    else:
        complaint = random.choice(list(COMPLAINT_ICD_MAPPING.keys()) + ADDITIONAL_COMPLAINTS)
    
    # Introduce local abbreviations and typos
    complaint = introduce_typos(complaint, typo_prob=0.15)
    
    # Generate ICD-10 code
    if complaint in COMPLAINT_ICD_MAPPING:
        icd_code = random.choice(COMPLAINT_ICD_MAPPING[complaint])
    else:
        icd_code = 'R69'  # Unknown cause
    
    # Introduce ICD-10 errors
    if random.random() < 0.1:
        icd_code = random.choice(['A15.9', 'ZZZ', 'Unknown', ''])
    elif random.random() < 0.05:
        icd_code = np.nan
    
    return complaint, icd_code

def generate_triage():
    """Generate triage category with typos"""
    categories = ['Red', 'Orange', 'Yellow', 'Green', 'Blue']
    weights = [0.05, 0.15, 0.30, 0.45, 0.05]  # Mostly Yellow/Green
    
    category = random.choices(categories, weights=weights)[0]
    
    # Introduce typos
    typos = {
        'Red': ['Red+', 'Read', 'Rede'],
        'Orange': ['Oringe', 'Ornage', 'Orang'],
        'Yellow': ['Yello', 'Yelow', 'Yeloo'],
        'Green': ['Greeen', 'Gren', 'Grean'],
        'Blue': ['Blew', 'Blu', 'Bloue']
    }
    
    if category in typos and random.random() < 0.15:
        category = random.choice(typos[category])
    
    return category

def generate_department(complaint, triage):
    """Generate department based on complaint and triage"""
    if 'antenatal' in complaint.lower() or 'maternity' in complaint.lower():
        dept = 'Antenatal'
    elif 'child' in complaint.lower() or (complaint in ['Cough & fever', 'child not feeding'] and random.random() < 0.7):
        dept = 'Paediatrics'
    elif triage in ['Red', 'Orange', 'Red+', 'Oringe']:
        dept = random.choice(['Emergency', 'Trauma'])
    elif 'TB' in complaint or 'HIV' in complaint:
        dept = 'HIV/TB Clinic'
    elif 'diabetes' in complaint or 'hypertension' in complaint:
        dept = random.choice(['OPD', 'Pharmacy'])
    else:
        dept = random.choice(list(DEPARTMENTS.keys()))
    
    # Get department variation
    if dept in DEPARTMENTS and random.random() < 0.4:
        dept = random.choice(DEPARTMENTS[dept])
    
    return introduce_typos(dept, typo_prob=0.1)

def generate_arrival_datetime():
    """Generate arrival datetime with messy formats"""
    # Generate random datetime between 2022-2023
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    random_date = start_date + timedelta(
        seconds=random.randint(0, int((end_date - start_date).total_seconds()))
    )
    
    # Different messy formats
    formats = [
        lambda d: d.strftime("%Y-%m-%d %H:%M"),  # 2023-05-15 14:30
        lambda d: d.strftime("%d/%m/%y %I.%M %p").lower(),  # 15/05/23 2.30 pm
        lambda d: d.strftime("%d-%b-%Y %Hh%M"),  # 15-May-2023 14h30
        lambda d: d.strftime("%Y%m%d %H%M"),  # 20230515 1430
        lambda d: d.strftime("%d/%m/%Y %H:%M")  # 15/05/2023 14:30
    ]
    
    # Sometimes missing date or time
    if random.random() < 0.03:
        return np.nan
    elif random.random() < 0.02:
        return random_date.strftime("%Y-%m-%d")  # Date only
    elif random.random() < 0.02:
        return random_date.strftime("%H:%M")  # Time only
    
    return random.choice(formats)(random_date)

def get_season(date_str):
    """Determine season from date string"""
    try:
        # Try to parse different date formats
        for fmt in ["%Y-%m-%d %H:%M", "%d/%m/%y %I.%M %p", "%d-%b-%Y %Hh%M", 
                   "%Y%m%d %H%M", "%d/%m/%Y %H:%M"]:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                break
            except:
                continue
        else:
            return random.choice(['Summer', 'Autumn', 'Winter', 'Spring'])
        
        month = date_obj.month
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        else:
            return 'Spring'
    except:
        return random.choice(['Summer', 'Autumn', 'Winter', 'Spring'])

def calculate_waiting_time(triage, department, arrival_time, facility_type, day_of_week):
    """Calculate realistic waiting time based on multiple factors"""
    base_time = random.randint(30, 480)  # Base 30-480 minutes
    
    # Triage impact
    if triage in ['Red', 'Red+']:
        modifier = 0.1  # -90%
    elif triage in ['Orange', 'Oringe']:
        modifier = 0.3  # -70%
    elif triage in ['Yellow', 'Yello']:
        modifier = 0.8
    elif triage in ['Green', 'Greeen']:
        modifier = 1.2
    else:  # Blue
        modifier = 1.0
    
    # Department impact
    if any(word in department for word in ['Emergency', 'ER', 'Emergncy', 'Trauma']):
        modifier *= 0.7  # -30%
    elif any(word in department for word in ['Pharmacy', 'Dispensary', 'OPD', 'Outpatient']):
        modifier *= 2.5  # +150%
    elif any(word in department for word in ['Clinic', 'CHC']):
        modifier *= 1.5  # +50%
    
    # Time of day impact (10-18h is peak)
    try:
        hour = int(arrival_time.split()[1][:2]) if ' ' in arrival_time else 12
        if 10 <= hour <= 18:
            modifier *= 1.4  # +40%
    except:
        pass
    
    # Weekend impact
    if day_of_week in ['Sat', 'Saturday', 'Sun', 'Sunday']:
        modifier *= 1.2  # +20%
    
    # Academic hospital impact
    if any(word in facility_type for word in ['ACAD', 'Academic', 'Academic Hosp']):
        modifier *= 1.25  # +25%
    
    # Add ±30% noise
    noise = random.uniform(0.7, 1.3)
    modifier *= noise
    
    waiting_time = int(base_time * modifier)
    
    # 2% extreme cases
    if random.random() < 0.01:
        return random.randint(1, 5)  # Very short
    elif random.random() < 0.01:
        return random.randint(1000, 2000)  # Very long
    
    return max(1, waiting_time)  # Ensure at least 1 minute

def generate_outcome(waiting_time, triage, department, age):
    """Generate outcome based on waiting time and other factors"""
    if waiting_time > 1000:  # Extreme wait
        outcomes = ['Absconded', 'Absconded', 'Treated and Discharged']
    elif triage in ['Red', 'Red+']:
        outcomes = ['Admitted to Ward', 'Admitted to Ward', 'Transferred', 'Died in ER']
    elif 'Pharmacy' in department or 'OPD' in department:
        outcomes = ['Treated and Discharged', 'Treated and Discharged', 'Referred to Specialist']
    elif age <= 5:
        outcomes = ['Treated and Discharged', 'Admitted to Ward', 'Referred to Specialist']
    elif waiting_time > 240:  # Long wait
        outcomes = ['Absconded', 'Treated and Discharged', 'Treated and Discharged']
    else:
        outcomes = ['Treated and Discharged', 'Admitted to Ward', 'Referred to Specialist', 'Transferred']
    
    return random.choice(outcomes)

def generate_advanced_columns(age, race, facility):
    """Generate optional advanced columns"""
    # Insurance status
    insurance_options = ['Medical Aid', 'Out-of-Pocket', 'Gov Subsidy', 'Unknown']
    insurance_weights = [0.15, 0.25, 0.55, 0.05]
    insurance = random.choices(insurance_options, weights=insurance_weights)[0]
    
    # Employment status
    if age < 18:
        employment = 'Student'
    elif age > 65:
        employment = 'Retired'
    else:
        employment_options = ['Employed', 'Unemployed', 'Informal']
        employment_weights = [0.35, 0.45, 0.20]
        employment = random.choices(employment_options, weights=employment_weights)[0]
    
    # Urban/Rural
    urban_rural = 'Urban' if any(word in facility for word in ['Soweto', 'CPT', 'DBN', 'Gqeberha', 'Hospital']) else 'Rural'
    
    # Follow-up flag
    follow_up_options = ['Follow-up scheduled', 'No follow-up', 'Re-admission within 30 days']
    follow_up_weights = [0.4, 0.55, 0.05]
    follow_up = random.choices(follow_up_options, weights=follow_up_weights)[0]
    
    return insurance, employment, urban_rural, follow_up

def generate_clinical_vitals(triage_category, age):
    """Generate realistic clinical vital signs based on triage category"""
    if pd.isna(triage_category) or triage_category not in CLINICAL_SEVERITY:
        return np.nan, np.nan, np.nan, np.nan
    
    severity = CLINICAL_SEVERITY[triage_category]
    
    # SATS score
    sats_score = random.randint(severity['min'], severity['max'])
    
    # Vital signs with age adjustments
    if age <= 1:  # Infant
        resp_rate = random.randint(30, 60)
        heart_rate = random.randint(100, 160)
    elif age <= 5:  # Toddler
        resp_rate = random.randint(20, 40)
        heart_rate = random.randint(80, 140)
    else:  # Adult/child
        resp_rate = random.randint(severity['resp_rate'][0], severity['resp_rate'][1])
        heart_rate = random.randint(severity['heart_rate'][0], severity['heart_rate'][1])
    
    # Oxygen saturation
    oxygen_saturation = random.randint(severity['sats'][0], severity['sats'][1])
    
    # Introduce some errors
    if random.random() < 0.03:
        resp_rate = random.choice([0, 999])
    if random.random() < 0.03:
        heart_rate = random.choice([0, 999])
    
    return sats_score, resp_rate, heart_rate, oxygen_saturation

def generate_geospatial_data(province, facility):
    """Generate realistic geographical coordinates"""
    if province not in PROVINCE_COORDS:
        return np.nan, np.nan
    
    coords = PROVINCE_COORDS[province]
    variation = coords['variation']
    
    # Add random variation around province centroid
    lat = coords['lat'] + random.uniform(-variation, variation)
    lon = coords['lon'] + random.uniform(-variation, variation)
    
    return lat, lon

def get_facility_capacity(facility_name):
    """Get facility capacity level"""
    # Clean facility name for matching
    clean_name = facility_name.split('(')[0].strip() if '(' in facility_name else facility_name
    for facility, capacity in FACILITY_CAPACITY.items():
        if clean_name in facility or facility in clean_name:
            return capacity
    return 'Unknown'

def get_staffing_level(arrival_datetime, facility_capacity):
    """Determine staffing level based on time and facility capacity"""
    if pd.isna(arrival_datetime):
        return 'Normal'
    
    try:
        # Parse datetime
        dt = pd.to_datetime(arrival_datetime, errors='coerce')
        if pd.isna(dt):
            return 'Normal'
        
        hour = dt.hour
        day = dt.weekday()
        
        # Night shift (7 PM to 7 AM)
        if hour < 7 or hour >= 19:
            return 'Reduced'
        # Weekend
        elif day >= 5:
            return 'Reduced'
        # Low capacity facility
        elif facility_capacity in ['Low', 'Very Low']:
            return 'Limited'
        else:
            return 'Normal'
    except:
        return 'Normal'

def introduce_more_data_quality_issues(df):
    """Introduce additional realistic data quality problems"""
    # Random null values in various columns
    for col in ['Gender', 'Race_Demographic', 'Triage_Category']:
        mask = np.random.random(len(df)) < 0.03
        df.loc[mask, col] = np.nan
    
    # Inconsistent date formats
    date_samples = df['Arrival_Date_Time'].dropna().sample(frac=0.1).index
    for idx in date_samples:
        try:
            dt = datetime.strptime(df.loc[idx, 'Arrival_Date_Time'], "%Y-%m-%d %H:%M")
            new_format = random.choice([
                lambda d: d.strftime("%d/%m/%Y %H:%M"),
                lambda d: d.strftime("%d-%b-%Y %Hh%M"),
                lambda d: d.strftime("%Y%m%d %H%M")
            ])
            df.loc[idx, 'Arrival_Date_Time'] = new_format(dt)
        except:
            pass
    
    return df

def generate_patient_journeys(df):
    """Create linked patient journeys across multiple visits"""
    # Create visit sequences for duplicate patients
    df['Visit_Number'] = 1
    duplicate_mask = df.duplicated('Patient_ID', keep=False)
    
    if duplicate_mask.any():
        duplicate_patients = df[duplicate_mask].copy()
        for patient_id in duplicate_patients['Patient_ID'].unique():
            patient_visits = duplicate_patients[duplicate_patients['Patient_ID'] == patient_id]
            if len(patient_visits) > 1:
                visit_nums = list(range(1, len(patient_visits) + 1))
                df.loc[df['Patient_ID'] == patient_id, 'Visit_Number'] = visit_nums
    
    return df

def validate_dataset(df):
    """Perform data validation and quality checks"""
    validation_results = {}
    
    # Check for invalid ages
    invalid_ages = df[(df['Age'] < 0) | (df['Age'] > 120)]
    validation_results['invalid_ages'] = len(invalid_ages)
    
    # Check waiting time outliers
    extreme_waits = df[df['Waiting_Time_Minutes'] > 1000]
    validation_results['extreme_wait_times'] = len(extreme_waits)
    
    # Check data completeness
    completeness = df.isnull().mean().to_dict()
    validation_results['completeness'] = completeness
    
    return validation_results

def generate_enhanced_dataset():
    """Generate the complete enhanced dataset"""
    data = []
    
    # Generate patient IDs first to ensure duplicates
    patient_ids = generate_patient_id()
    
    for i in range(NUM_RECORDS):
        if i % 50000 == 0:
            print(f"Generating record {i}/{NUM_RECORDS}...")
        
        # Basic demographics
        patient_id = patient_ids[i]
        age = generate_age()
        gender = generate_gender()
        race = generate_race()
        province, facility = generate_province_facility()
        
        # Temporal data
        arrival_datetime = generate_arrival_datetime()
        season = get_season(arrival_datetime) if pd.notna(arrival_datetime) else random.choice(['Summer', 'Autumn', 'Winter', 'Spring'])
        day_of_week = random.choice(['Mon', 'Monday', 'Tue', 'tuesday', 'Wed', 'Wednesday', 
                                   'Thu', 'Thursday', 'Fri', 'Friday', 'Sat', 'Saturday', 
                                   'Sun', 'Sunday'])
        
        # Medical data
        complaint, icd_code = generate_complaint_and_icd(age, gender, race, season)
        triage = generate_triage()
        department = generate_department(complaint, triage)
        
        # Calculate waiting time
        waiting_time = calculate_waiting_time(triage, department, arrival_datetime, facility, day_of_week)
        
        # Outcome
        outcome = generate_outcome(waiting_time, triage, department, age)
        
        # Advanced columns
        insurance, employment, urban_rural, follow_up = generate_advanced_columns(age, race, facility)
        
        # New enhanced columns
        sats_score, resp_rate, heart_rate, oxygen_sat = generate_clinical_vitals(triage, age)
        latitude, longitude = generate_geospatial_data(province, facility)
        facility_capacity = get_facility_capacity(facility)
        staffing_level = get_staffing_level(arrival_datetime, facility_capacity)
        
        # Facility referral ID (track referrals)
        referral_id = f"REF-{random.randint(10000, 99999)}" if random.random() < 0.3 else np.nan
        
        data.append({
            'Patient_ID': patient_id,
            'Province': province,
            'Facility_Name': facility,
            'Age': age,
            'Gender': gender,
            'Race_Demographic': race,
            'Chief_Complaint': complaint,
            'ICD-10_Code': icd_code,
            'Triage_Category': triage,
            'Department': department,
            'Arrival_Date_Time': arrival_datetime,
            'Day_of_Week': day_of_week,
            'Season': season,
            'Waiting_Time_Minutes': waiting_time,
            'Outcome': outcome,
            'Insurance_Status': insurance,
            'Employment_Status': employment,
            'Urban_Rural': urban_rural,
            'Follow_Up_Flag': follow_up,
            # Enhanced columns
            'SATS_Score': sats_score,
            'Respiratory_Rate': resp_rate,
            'Heart_Rate': heart_rate,
            'Oxygen_Saturation': oxygen_sat,
            'Latitude': latitude,
            'Longitude': longitude,
            'Facility_Capacity': facility_capacity,
            'Staffing_Level': staffing_level,
            'Facility_Referral_ID': referral_id
        })
    
    df = pd.DataFrame(data)
    
    # Apply additional enhancements
    df = introduce_more_data_quality_issues(df)
    df = generate_patient_journeys(df)
    
    return df

def save_to_multiple_formats(df):
    """Save the dataset in multiple formats"""
    # Create output directory if it doesn't exist
    os.makedirs('healthcare_data_output', exist_ok=True)
    
    # CSV format
    csv_path = "healthcare_data_output/enhanced_south_african_healthcare_dataset.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Saved CSV to {csv_path}")
    
    # JSON format
    json_path = "healthcare_data_output/enhanced_south_african_healthcare_dataset.json"
    df.to_json(json_path, orient='records', indent=2)
    print(f"Saved JSON to {json_path}")
    
    # Parquet format
    parquet_path = "healthcare_data_output/enhanced_south_african_healthcare_dataset.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet to {parquet_path}")
    
    # Excel format (sample of first 1000 rows to avoid large file)
    excel_path = "healthcare_data_output/enhanced_south_african_healthcare_dataset_sample.xlsx"
    df.head(1000).to_excel(excel_path, index=False)
    print(f"Saved Excel sample to {excel_path}")
    
    return csv_path

def create_sqlite_database(csv_path):
    """Create SQLite database and import data"""
    try:
        # Connect to SQLite database
        conn = sqlite3.connect('healthcare_data_output/south_african_healthcare.db')
        cursor = conn.cursor()
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Clean column names for SQL compatibility
        df.columns = [col.replace('-', '_').replace(' ', '_') for col in df.columns]
        
        # Create table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS patient_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Patient_ID TEXT,
            Province TEXT,
            Facility_Name TEXT,
            Age INTEGER,
            Gender TEXT,
            Race_Demographic TEXT,
            Chief_Complaint TEXT,
            ICD_10_Code TEXT,
            Triage_Category TEXT,
            Department TEXT,
            Arrival_Date_Time TEXT,
            Day_of_Week TEXT,
            Season TEXT,
            Waiting_Time_Minutes INTEGER,
            Outcome TEXT,
            Insurance_Status TEXT,
            Employment_Status TEXT,
            Urban_Rural TEXT,
            Follow_Up_Flag TEXT,
            SATS_Score INTEGER,
            Respiratory_Rate INTEGER,
            Heart_Rate INTEGER,
            Oxygen_Saturation INTEGER,
            Latitude REAL,
            Longitude REAL,
            Facility_Capacity TEXT,
            Staffing_Level TEXT,
            Facility_Referral_ID TEXT,
            Visit_Number INTEGER
        )
        """
        cursor.execute(create_table_query)
        
        # Insert data
        df.to_sql('patient_records', conn, if_exists='replace', index=False)
        
        # Verify row count
        cursor.execute("SELECT COUNT(*) FROM patient_records")
        sqlite_count = cursor.fetchone()[0]
        
        print(f"SQLite database created with {sqlite_count} records")
        
        # Create indexes for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patient_id ON patient_records (Patient_ID)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_province ON patient_records (Province)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_triage ON patient_records (Triage_Category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_department ON patient_records (Department)")
        
        # Close connection
        conn.close()
        
        return sqlite_count
        
    except Error as e:
        print(f"SQLite error: {e}")
        return 0

def create_mongodb_database(csv_path):
    """Create MongoDB database and import data with better error handling"""
    try:
        # Try to connect to MongoDB
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        
        # Test the connection
        client.admin.command('ismaster')
        
        db = client["south_african_healthcare"]
        collection = db["patient_records"]
        
        # Clear existing collection
        collection.delete_many({})
        
        # Read and prepare data
        df = pd.read_csv(csv_path)
        
        # Convert to dictionary for MongoDB
        records = df.to_dict('records')
        
        # Insert data in batches
        batch_size = 10000
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            collection.insert_many(batch)
            print(f"Inserted batch {i//batch_size + 1}/{(len(records)//batch_size)+1}")
        
        # Verify row count
        mongo_count = collection.count_documents({})
        print(f"MongoDB database created with {mongo_count} records")
        
        # Close connection
        client.close()
        
        return mongo_count
        
    except pymongo.errors.ServerSelectionTimeoutError:
        print("MongoDB is not running. Skipping MongoDB import.")
        return 0
    except Exception as e:
        print(f"MongoDB error: {e}")
        return 0

def create_summary_report(df, csv_count, sqlite_count, mongo_count):
    """Create a comprehensive summary report of the dataset"""
    report = {
        "dataset_overview": {
            "total_records": int(len(df)),
            "unique_patients": int(df['Patient_ID'].nunique()),
            "duplicate_records": int(len(df) - df['Patient_ID'].nunique()),
            "data_generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "demographics": {
            "age_statistics": {
                "min": float(df['Age'].min()) if not pd.isna(df['Age'].min()) else None,
                "max": float(df['Age'].max()) if not pd.isna(df['Age'].max()) else None,
                "mean": float(df['Age'].mean()) if not pd.isna(df['Age'].mean()) else None,
                "null_count": int(df['Age'].isnull().sum())
            },
            "gender_distribution": {k: int(v) for k, v in df['Gender'].value_counts().to_dict().items()},
            "race_distribution": {k: int(v) for k, v in df['Race_Demographic'].value_counts().to_dict().items()}
        },
        "clinical_metrics": {
            "average_waiting_time": float(df['Waiting_Time_Minutes'].mean()),
            "triage_distribution": {k: int(v) for k, v in df['Triage_Category'].value_counts().to_dict().items()},
            "outcome_distribution": {k: int(v) for k, v in df['Outcome'].value_counts().to_dict().items()}
        },
        "facility_analysis": {
            "province_distribution": {k: int(v) for k, v in df['Province'].value_counts().to_dict().items()},
            "facility_capacity_distribution": {k: int(v) for k, v in df['Facility_Capacity'].value_counts().to_dict().items()}
        },
        "data_quality": {
            "null_rates": {k: float(v) for k, v in df.isnull().mean().to_dict().items()},
            "invalid_ages": int(len(df[(df['Age'] < 0) | (df['Age'] > 120)])),
            "extreme_wait_times": int(len(df[df['Waiting_Time_Minutes'] > 1000]))
        },
        "database_import": {
            "csv_records": int(csv_count),
            "sqlite_records": int(sqlite_count),
            "mongodb_records": int(mongo_count),
            "all_databases_match": bool(csv_count == sqlite_count == mongo_count)
        }
    }
    
    # Save the report as JSON
    report_path = "healthcare_data_output/dataset_summary_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Summary report saved to {report_path}")
    return report

def create_visualizations(df):
    """Create basic visualizations for the dataset"""
    try:
        # Set style
        sns.set_style("whitegrid")
        plt.figure(figsize=(15, 10))
        
        # Create output directory for visualizations
        os.makedirs('healthcare_data_output/visualizations', exist_ok=True)
        
        # 1. Triage category distribution
        plt.subplot(2, 2, 1)
        df['Triage_Category'].value_counts().plot(kind='bar', color='skyblue')
        plt.title('Triage Category Distribution')
        plt.xticks(rotation=45)
        
        # 2. Waiting time distribution
        plt.subplot(2, 2, 2)
        # Filter out extreme values for better visualization
        filtered_waiting = df[df['Waiting_Time_Minutes'] <= 1000]['Waiting_Time_Minutes']
        plt.hist(filtered_waiting, bins=50, color='lightgreen', edgecolor='black')
        plt.title('Waiting Time Distribution (≤ 1000 minutes)')
        plt.xlabel('Minutes')
        
        # 3. Province distribution
        plt.subplot(2, 2, 3)
        df['Province'].value_counts().plot(kind='bar', color='lightcoral')
        plt.title('Patient Distribution by Province')
        plt.xticks(rotation=45)
        
        # 4. Outcome distribution
        plt.subplot(2, 2, 4)
        df['Outcome'].value_counts().plot(kind='bar', color='gold')
        plt.title('Outcome Distribution')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('healthcare_data_output/visualizations/dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional visualizations
        # Age distribution
        plt.figure(figsize=(10, 6))
        df['Age'].hist(bins=50, color='lightblue', edgecolor='black')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Count')
        plt.savefig('healthcare_data_output/visualizations/age_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Facility capacity by province
        plt.figure(figsize=(12, 8))
        capacity_by_province = pd.crosstab(df['Province'], df['Facility_Capacity'])
        capacity_by_province.plot(kind='bar', stacked=True)
        plt.title('Facility Capacity Distribution by Province')
        plt.xticks(rotation=45)
        plt.legend(title='Facility Capacity')
        plt.tight_layout()
        plt.savefig('healthcare_data_output/visualizations/capacity_by_province.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations created and saved to healthcare_data_output/visualizations/")
        
    except ImportError:
        print("Matplotlib or Seaborn not installed. Skipping visualizations.")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def main():
    """Main function to generate data and load into databases"""
    # Generate the enhanced dataset
    print("Generating enhanced South African public healthcare dataset...")
    enhanced_df = generate_enhanced_dataset()
    
    # Save to multiple formats
    csv_path = save_to_multiple_formats(enhanced_df)
    
    # Display sample of the data
    print("\nSample of the enhanced dataset:")
    print(enhanced_df.head(10).to_string(index=False))
    
    # Display validation results
    validation = validate_dataset(enhanced_df)
    print(f"\nValidation Results:")
    print(f"Invalid ages: {validation['invalid_ages']}")
    print(f"Extreme wait times: {validation['extreme_wait_times']}")
    print(f"Data completeness (null rates):")
    for col, null_rate in validation['completeness'].items():
        print(f"  {col}: {null_rate:.3%}")
    
    # Display some statistics
    print(f"\nDataset Statistics:")
    print(f"Total records: {len(enhanced_df)}")
    print(f"Unique patients: {enhanced_df['Patient_ID'].nunique()}")
    print(f"Duplicate patient records: {len(enhanced_df) - enhanced_df['Patient_ID'].nunique()}")
    print(f"Average waiting time: {enhanced_df['Waiting_Time_Minutes'].mean():.1f} minutes")
    print(f"Most common triage category: {enhanced_df['Triage_Category'].mode()[0] if not enhanced_df['Triage_Category'].mode().empty else 'N/A'}")
    print(f"Facility capacity distribution:")
    print(enhanced_df['Facility_Capacity'].value_counts())
    
    # Load data into databases
    print("\nLoading data into databases...")
    
    # SQLite
    sqlite_count = create_sqlite_database(csv_path)
    
    # MongoDB
    mongo_count = create_mongodb_database(csv_path)
    
    # Verify row counts
    csv_count = len(enhanced_df)
    print(f"\nRow count verification:")
    print(f"CSV file: {csv_count} records")
    print(f"SQLite database: {sqlite_count} records")
    print(f"MongoDB database: {mongo_count} records")
    
    if csv_count == sqlite_count == mongo_count:
        print("✓ All row counts match!")
    else:
        print("✗ Row counts do not match!")
    
    # Create summary report
    print("\nCreating summary report...")
    report = create_summary_report(enhanced_df, csv_count, sqlite_count, mongo_count)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(enhanced_df)
    
    print("\nData generation and loading completed successfully!")
    
    # Display MongoDB instructions if not connected
    if mongo_count == 0:
        print("\n" + "="*60)
        print("MONGODB SETUP INSTRUCTIONS:")
        print("To enable MongoDB integration, please:")
        print("1. Install MongoDB on your system")
        print("2. Start the MongoDB service")
        print("3. Run this script again")
        print("="*60)

if __name__ == "__main__":
    main()