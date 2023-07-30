# src/config/column_names.py

# Basic columns
BASIC_COLUMNS = {
    "numeric": [
        "AGE",  # Age of the recipient at the time of the transplant
        "AGE_DON",  # Age of the donor at the time of the donation
        "BMI_CALC",  # Body Mass Index (BMI) of the recipient calculated as weight (kg) / height (m^2)
        "BMI_DON_CALC",  # Body Mass Index (BMI) of the donor calculated as weight (kg) / height (m^2)
        "DAYSWAIT_CHRON_KI",  # The number of days the recipient waited on the kidney transplant list
        "COLD_ISCH_KI",  # Kidney cold ischemic time (hours)
        "KDRI_RAO",  # Kidney Donor Risk Index (risk score)
        "CREAT_TRR",  # Creatinine level of the recipient at the time of transplant 0.201259
        "DIAL_LEN", # Length of time on dialysis (days)
    ],
    "categorical": [
        "ETHCAT",  # Ethnic category of the recipient
        "ETHCAT_DON",  # Ethnic category of the donor
        "GENDER",  # Gender of the recipient
        "GENDER_DON",  # Gender of the donor
        "ABO_MAT",  # Blood type match between the recipient and the donor
        "AMIS",  # A locus mismatch score between the recipient and the donor
        "BMIS",  # B locus mismatch score between the recipient and the donor
        "DRMIS",  # DR locus mismatch score between the recipient and the donor
        "DIAB",  # Diabetes status of the recipient
        "DON_TY",  # Type of donor (living, deceased, etc.)
        "DIAL_TRR",  # Dialysis status of the recipient at the time of transplant
        "ON_DIALYSIS",  # Recipient on dialysis before transplant (yes/no)
        "REGION",  # Region of the transplant center
        "HIST_HYPERTENS_DON",  # Donor history of hypertension rsf = 0.0003973392811658039
        "HCV_SEROSTATUS",
        "DEATH_MECH_DON",
        "DIAG_KI",
        "FUNC_STAT_TRR",
        "COD_CAD_DON",
    ]
}

# Extra columns
EXTRA_COLUMNS = {
    "numeric": [
        "CREAT_DON",  # DECEASED DONOR-TERMINAL LAB CREATININE 0.362306
        "KDRI_MED",  # Kidney Donor Risk Index (risk score)
        "KDPI",  # Kidney Donor Profile Index (risk score)
        "DAYSWAIT_ALLOC",  # TIME USED FOR ALLOCATION PRIORITY (DAYS)
        "CURRENT_PRA",  # Candidate most recent "current" PRA from waiting list/allocation
        "INIT_CPRA",  # CANDIDATE CALCULATED PRA AT LISTING
        "END_CPRA",  # Calculated Panel Reactive Antibody (cPRA) value at the end of the waiting period percent_null:0.426746 rsf = 3.858056993504093e-05
    ],
    "categorical": [
        "HLAMIS",  # HLA locus mismatch score between the recipient and the donor
        "DRUGTRT_COPD",  # Recipient drug treated COPD at registration rsd = 1.77048445436645e-06
        "HCV_SEROSTATUS",  # Recipient Hepatitis C Virus serostatus rsf=-2.3015112276314475e-06
        "MED_COND_TRR",  #  Recipient medical condition pre-transplant at transplant HOSPITALIZED NOT IN ICU, IN INTENSIVE CARE UNIT, NOT HOSPITALIZED
        "PRE_TX_TXFUS",  # RECIPIENT # OF PRE-TRANSPLANT TRANSFUSIONS-KI  @ TRANSPLANT rsf = 1.0657349506985417e-05
        "DGN_TCR",  # PRIMARY DIAGNOSIS AT TIME OF LISTING
        "COD_CAD_DON",  # Deceased donor cause of death
        "EBV_SEROSTATUS",  # Recipient Epstein-Barr Virus serostatus
        "CMV_STATUS",  # Recipient Cytomegalovirus serostatus
        "ECD_DONOR",  # Extended criteria donor (yes/no)
        "ABO",  # Recipient blood group at registration
        "ABO_DON",  # Donor blood type
        "DEATH_MECH_DON",  # Mechanism of death of the donor
        "FUNC_STAT_TRR",  # Functional status of the recipient at the time of transplant
        "CMV_DON",  # Donor serology anti-CMV
        "CITIZENSHIP",  # Citizenship of the candidate at registration
        "CITIZENSHIP_DON",  # Donor citizenship
        "FUNC_STAT_TCR",  # Functional status at the time of registration 1.8 % missing
        "MED_COND_TRR",
        "PRI_PAYMENT_TRR_KI",  # RECIPIENT PRIMARY PAYMENT SOURCE-KI  @ TRANSPLANT
        "DIAG_KI",  # Primary diagnosis of the recipent has to be grouped too many categories
    ]
}

# Output variables
OUTPUT_VARS = {
    "kidney": [
        "GSTATUS_KI",  # Graft status of the kidney transplant (successful or failed)
        "GTIME_KI",  # Time to graft failure or last follow-up
    ],
    "death": [
        "PSTATUS",  # Boolean Most Recent Patient Status (based on composite death date) (1=Dead, 0=Alive)
        "PTIME",  # Patient Survival Time in days (based on composite death date)
    ],
}

