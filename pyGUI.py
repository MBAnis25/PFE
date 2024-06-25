import pandas as pd
import numpy as np
import io
import streamlit as st
import matplotlib.pyplot as plt
import re

# --- Functions --- 
def calculate_hi_and_assign_zones(data):
    """
    Calculates Heterogeneity Indices (HI) for oil and water production,
    assigns wells to zones based on cumulative HI values, 
    and identifies wells close to the origin (Zone 5).
    """
    data['DaysinMonth'] = pd.to_datetime(data['Date']).dt.days_in_month
    data['daily Oil rate'] = data['Oil'] / data['DaysinMonth']
    data['daily Water rate'] = data['Water'] / data['DaysinMonth']
    data['Group average Oil rate'] = data.groupby('Date')['daily Oil rate'].transform(lambda x: x[x > 0].mean())
    data['Group average Water rate'] = data.groupby('Date')['daily Water rate'].transform(lambda x: x[x > 0].mean())
    data['HI oil'] = (data['daily Oil rate'] / data['Group average Oil rate']) - 1
    data['HI water'] = (data['daily Water rate'] / data['Group average Water rate']) - 1
    data['Cum HI oil'] = data.groupby('Well')['HI oil'].cumsum()
    data['Cum HI water'] = data.groupby('Well')['HI water'].cumsum()
    data = data.groupby('Well').agg({'Cum HI oil': 'last', 'Cum HI water': 'last'}).reset_index()
    data["Cum HI oil"] = pd.to_numeric(data["Cum HI oil"], errors='coerce')
    data["Cum HI water"] = pd.to_numeric(data["Cum HI water"], errors='coerce')

    conditions = [
        (data["Cum HI oil"] > 0) & (data["Cum HI water"] > 0),
        (data["Cum HI oil"] < 0) & (data["Cum HI water"] < 0),
        (data["Cum HI oil"] < 0) & (data["Cum HI water"] > 0),
        (data["Cum HI oil"] > 0) & (data["Cum HI water"] < 0)
    ]
    choices = ["1", "2", "3", "4"]
    data['Zone'] = np.select(conditions, choices, default='Zone Unknown')

    data["distance_from_zero"] = np.sqrt(data["Cum HI oil"]**2 + data["Cum HI water"]**2)
    num_actions_zone5 = int(0.15 * len(data))
    zone5_actions = data.nsmallest(num_actions_zone5, "distance_from_zero")["Well"]
    data.loc[data["Well"].isin(zone5_actions), "Zone"] = "5"

    data = data.drop("distance_from_zero", axis=1)
    return data

def calculate_normalized_weighted_values(data, weights):
    """Normalizes criteria data and applies weights."""
    weighted_normalized_data = (data / (np.sqrt(np.sum(data**2, axis=0)))) * weights
    return weighted_normalized_data

def calculate_ideal_and_anti_ideal_points(weighted_data, criteria_types):
    """
    Calculates the ideal and anti-ideal points based on
    whether each criterion should be maximized or minimized.
    """
    ideal = []
    anti_ideal = []
    for idx, maximize in enumerate(criteria_types):
        if maximize:
            ideal.append(np.max(weighted_data[:, idx]))
            anti_ideal.append(np.min(weighted_data[:, idx]))
        else:
            ideal.append(np.min(weighted_data[:, idx]))
            anti_ideal.append(np.max(weighted_data[:, idx]))
    return np.array(ideal), np.array(anti_ideal)

def calculate_distances_and_scores(weighted_data, ideal_point, anti_ideal_point):
    """
    Calculates distances to ideal and anti-ideal points,
    and computes TOPSIS scores.
    """
    distances_to_ideal = np.sqrt(np.sum((weighted_data - ideal_point)**2, axis=1))
    distances_to_anti_ideal = np.sqrt(np.sum((weighted_data - anti_ideal_point)**2, axis=1))
    epsilon = 1e-10 
    scores = distances_to_anti_ideal / (distances_to_ideal + distances_to_anti_ideal + epsilon)
    scores = np.round(scores, 5)
    return distances_to_ideal, distances_to_anti_ideal, scores

def calculer_vecteur_priorite(matrice):
    """Calculates the priority vector from an AHP comparison matrix."""
    somme_colonnes = np.sum(matrice, axis=0)
    matrice_normalisee = matrice / somme_colonnes
    vecteur_priorite = np.mean(matrice_normalisee, axis=1)
    vecteur_priorite_arrondi = np.round(vecteur_priorite, 3)
    return vecteur_priorite_arrondi

# --- ELECTRE Functions ---

def calculate_concordance_matrix(scores, weights):
    """Calculates the concordance matrix for ELECTRE."""
    num_actions = scores.shape[0]
    concordance_matrix = np.zeros((num_actions, num_actions))

    for i in range(num_actions):
        for j in range(num_actions):
            if i != j:
                # Check if action i dominates action j for each criterion
                concordant_criteria = scores[i] >= scores[j]
                # Sum weights of concordant criteria
                concordance_matrix[i, j] = np.sum(weights[concordant_criteria])

    return concordance_matrix

def calculate_discordance_matrix(scores, weights, thresholds, scale_lengths):
    """Calculates the discordance matrix for ELECTRE."""
    scores = np.asarray(scores)
    scale_lengths = np.asarray(scale_lengths)
    num_actions = scores.shape[0]
    discordance_matrix = np.zeros((num_actions, num_actions))

    for k in range(num_actions):
        for i in range(num_actions):
            if k != i:
                differences = scores[k] - scores[i]
                # Find criteria where action k is significantly worse than action i
                discordant_criteria = np.where(differences < -thresholds)[0]
                
                for criterion in discordant_criteria:
                    # Calculate normalized difference
                    normalized_difference = abs(differences[criterion]) / scale_lengths[criterion]
                    # Update discordance matrix if normalized difference exceeds weight
                    if normalized_difference > weights[criterion]:
                        discordance_matrix[k, i] = 1
                        break

    return discordance_matrix

def calculate_outranking_relationships(concordance_matrix, discordance_matrix, concordance_threshold, discordance_threshold):
    """Determines outranking relationships based on thresholds."""
    num_actions = concordance_matrix.shape[0]
    outranking_matrix = np.zeros((num_actions, num_actions), dtype=bool)

    for i in range(num_actions):
        for j in range(num_actions):
            if i != j:
                if concordance_matrix[i, j] >= concordance_threshold and discordance_matrix[i, j] <= discordance_threshold:
                    outranking_matrix[i, j] = True

    return outranking_matrix

def rank_actions_electre(outranking_matrix, actions):
    """Ranks actions based on outranking relationships."""
    outranking_counts = {action: np.sum(outranking_matrix[i]) for i, action in enumerate(actions)}
    sorted_actions = sorted(outranking_counts.items(), key=lambda item: item[1], reverse=True)  # Sort in descending order
    return [action for action, _ in sorted_actions], outranking_counts


def validate_ahp_matrix(matrix):
    """
    Validates the AHP comparison matrix.
    - Diagonal values must be 1.
    - Values must be in the set [1, 2, 3, 4, 5, 6, 7, 8, 9] or their reciprocals.
    - Values above the diagonal must be the reciprocal of the values below the diagonal.
    """
    valid_values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9}


    num_rows, num_cols = matrix.shape

    for i in range(num_rows):
        if matrix[i, i] != 1:
            return False, f"Diagonal element at row {i+1}, column {i+1} is not 1."

        for j in range(num_cols):
            if matrix[i, j] not in valid_values:
                return False, f"Element at row {i+1}, column {j+1} is not a valid value."
            if matrix[j, i] != 0 and matrix[i, j] != 1 / matrix[j, i]:
                return False, f"Element at row {i+1}, column {j+1} is not the reciprocal of element at row {j+1}, column {i+1}."

    return True, "Matrix is valid."






# --- Main Script --- 
st.set_page_config(page_title="Well Optimization Analysis", layout="wide")

# --- Sidebar ---
st.sidebar.title("Navigation")

page = st.sidebar.radio("Select a page:", ["Home", "Features", "Instructions", "Analysis"])

if page == "Home":
    # --- Logos ---
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        st.image("Logo__USTHB.png")  # Replace with the actual path to your logo
    with col3:
        st.image("SLB_Logo_2022 (1).png")  # Replace with the actual path to your logo

    import streamlit as st

    # --- Title and Subtitle ---
    st.markdown("<h1 style='text-align: center;'>Automation of Work-Over Candidates Selection Using Heterogeneity Index</h1>", unsafe_allow_html=True)

    # Utilisez un titre plus petit pour le sous-titre
    st.markdown("<h2 style='text-align: center;'>Welcome to our Graphical Interface</h2>", unsafe_allow_html=True)

    # Utilisez un paragraphe pour le texte explicatif
    #st.markdown("<p style='text-align: center;'>This application helps you analyze and optimize well performance based on various criteria.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This application streamlines the process of selecting work-over candidate wells by categorizing them based on geographic zones.  It then prioritizes wells within each zone, helping you determine the optimal sequence for work-over operations.</p>", unsafe_allow_html=True)
    # Ajoutez un peu d'espace entre les sections
    st.markdown("---")
    
    # --- Developer Names and Supervisors (Right and Left Alignment) ---
    col1, col2, col3 = st.columns([20, 35, 20])
    with col1:  # Use the leftmost column
        st.markdown("Supervised by :")
        st.markdown("AMROUCHE Karim (USTHB)")
        st.markdown("HAMIDI Borhane (SLB)")
        

    with col3:  # Use the rightmost column
        st.markdown("By :")
        st.markdown("ABDELLIOUI Oussama")
        st.markdown("BOUKHALFA Mohamed Anis")


# --- Features Page ---
if page == "Features":
    # --- Features ---
    st.title("Features")
    st.write("**This application includes the following features:**")
    st.markdown("- Calculate Heterogeneity Indices (HI) for oil and water production")
    st.markdown("- Assign wells to zones based on cumulative HI values")
    st.markdown("- Perform TOPSIS and ELECTRE analyses to rank wells based on AHP weights")

# --- Instructions Page ---
if page == "Instructions":
    # --- Instructions ---
    st.title("Instructions")
    st.write("**Follow these steps to use the application:**")
    st.markdown("1. **Prepare your well data in an Excel file with the following worksheets:**")
    st.markdown("    - **Production:** Contains well production data (e.g., `Date`, `Well`, `Oil`, `Water`).")
    st.markdown("    - **Criteria:** Defines the evaluation criteria for work-over candidates.")
    st.markdown("    - **Zone 1 to Zone 5:**  Contains data for each zone (e.g., Pairwise Comparison Matrix , Criteria type).")
    st.markdown("2. **Upload the Excel file on the `Analysis` page.**")
    st.markdown("3. **The application will calculate Heterogeneity Indices and assign wells to zones.**")
    st.markdown("4. **The application will calculate and appear the criteria weights using the AHP method.**")
    st.markdown("5. **View and interpret the results to make informed decisions on well optimization.**")

# --- Analysis Page ---
if page == "Analysis":
    st.title("Well Optimization Analysis")

    # --- Sidebar ---
    st.title("Input & Zone Selection")

    # --- File Uploader ---
    uploaded_file = st.file_uploader("Upload your Excel file (DATA.xlsx)", type=['xlsx'])

    if uploaded_file is not None:
        # --- Read Excel Data ---
        try:
            sheets = pd.read_excel(uploaded_file, sheet_name=None)
            Prod = sheets['Prod']
            Crit = sheets['Crit']

            # --- Validate "Prod" Data ---
            # --- Validate "Prod" Data ---
            try:

                # Convert the 'Date' column to datetime objects
                Prod['Date'] = pd.to_datetime(Prod['Date'], format='%d-%b-%y')
            # Iterate through each well and check for ascending order in 'Date'
                
                # Group the DataFrame by 'Well'
                wells = Prod.groupby('Well')

                for well_name, well_data in wells:
                    # Check if 'Date' is in ascending order
                    for i in range(len(well_data['Date']) - 1):
                        if well_data['Date'].iloc[i] > well_data['Date'].iloc[i+1]:
                            st.error(f"Well '{well_name}' in 'Prod' sheet has dates out of order.")
                            st.write(f"Date {well_data['Date'].iloc[i]} is followed by earlier date {well_data['Date'].iloc[i+1]}")
                            raise ValueError("Please check your data.")
            
                
                
                    # --- Validate "Prod" Data ---
                    try:
                        # Check for negative values in Oil and Water columns
                        if (Prod['Oil'] < 0).any() or (Prod['Water'] < 0).any():
                            # Display the exact negative cases
                            st.error("Error in 'Prod' sheet: Negative values found in Oil or Water columns.")
                            negative_oil_rows = Prod[Prod['Oil'] < 0]
                            negative_water_rows = Prod[Prod['Water'] < 0]
                            if not negative_oil_rows.empty:
                                st.write("Negative Oil values:")
                                st.write(negative_oil_rows[['Date', 'Well', 'Oil']])
                            if not negative_water_rows.empty:
                                st.write("Negative Water values:")
                                st.write(negative_water_rows[['Date', 'Well', 'Water']])
                            raise ValueError("Please check your data.")
                        
                        # Check if well names in "Prod" and "Crit" sheets match the required format
                        well_name_pattern = re.compile(r'^\d{2}-\d{2}$')
                
                        invalid_well_names_prod = Prod[~Prod['Well'].astype(str).str.match(well_name_pattern)]
                        invalid_well_names_crit = Crit[~Crit['Well'].astype(str).str.match(well_name_pattern)]
                
                        if not invalid_well_names_prod.empty:
                            st.error("Error in 'Prod' sheet: Well names not in required format (xx-xx).")
                            st.write("Invalid Well names:")
                            st.write(invalid_well_names_prod[['Well']])
                
                        if not invalid_well_names_crit.empty:
                            st.error("Error in 'Crit' sheet: Well names not in required format (xx-xx).")
                            st.write("Invalid Well names:")
                            st.write(invalid_well_names_crit[['Well']])
                
                        if not invalid_well_names_prod.empty or not invalid_well_names_crit.empty:
                            raise ValueError("Please check your data.")
                            
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                
                # Check for negative values in Oil and Water columns
                if (Prod['Oil'] < 0).any() or (Prod['Water'] < 0).any():
                    # Display the exact negative cases
                    st.error("Error in 'Prod' sheet: Negative values found in Oil or Water columns.")
                    negative_oil_rows = Prod[Prod['Oil'] < 0]
                    negative_water_rows = Prod[Prod['Water'] < 0]
                    if not negative_oil_rows.empty:
                        st.write("Negative Oil values:")
                        st.write(negative_oil_rows[['Date', 'Well', 'Oil']])
                    if not negative_water_rows.empty:
                        st.write("Negative Water values:")
                        st.write(negative_water_rows[['Date', 'Well', 'Water']])
                    raise ValueError("Please check your data.")

                # Check if "SO" and "PHI" values are between 0 and 1
                if (Crit['SO'] < 0).any() or (Crit['SO'] > 1).any() or (Crit['PHI'] < 0).any() or (Crit['PHI'] > 1).any():
                    # Display the exact cases with invalid "SO" or "PHI" values
                    st.error("Error in 'Crit' sheet: 'SO' or 'PHI' values out of range (0 to 1).")
                    invalid_so_rows = Crit[(Crit['SO'] < 0) | (Crit['SO'] > 1)]
                    invalid_phi_rows = Crit[(Crit['PHI'] < 0) | (Crit['PHI'] > 1)]
                    if not invalid_so_rows.empty:
                        st.write("Invalid 'SO' values:")
                        st.write(invalid_so_rows[['Well', 'SO']])
                    if not invalid_phi_rows.empty:
                        st.write("Invalid 'PHI' values:")
                        st.write(invalid_phi_rows[['Well', 'PHI']])
                    raise ValueError("Please check your data.")
                  
                # Check well name format in "Prod" sheet
                well_format = re.compile(r'^\d{2}-\d{2}$')
                invalid_well_names_prod = Prod[~Prod['Well'].str.match(well_format)]
                if not invalid_well_names_prod.empty:
                    st.error("Error in 'Prod' sheet: Invalid well name format found.")
                    st.write("Invalid well names:")
                    st.write(invalid_well_names_prod[['Date', 'Well']])
                    raise ValueError("Please check the well name format (xx-xx) in 'Prod' sheet.")

                # Check well name format in "Crit" sheet
                invalid_well_names_crit = Crit[~Crit['Well'].str.match(well_format)]
                if not invalid_well_names_crit.empty:
                    st.error("Error in 'Crit' sheet: Invalid well name format found.")
                    st.write("Invalid well names:")
                    st.write(invalid_well_names_crit[['Well']])
                    raise ValueError("Please check the well name format (xx-xx) in 'Crit' sheet.")




                    # Check well name format in "Crit" sheet
                    invalid_well_names_crit = Crit[~Crit['Well'].str.match(well_format)]
                    if not invalid_well_names_crit.empty:
                        st.error("Error in 'Crit' sheet: Invalid well name format found.")
                        st.write("Invalid well names:")
                        st.write(invalid_well_names_crit[['Well']])
                        raise ValueError("Please check the well name format (xx-xx) in 'Crit' sheet.")

                
                # --- Validate "Zone" Data ---
                valid_zones = ["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5"]
                for zone in valid_zones:
                    if zone in sheets:
                        zone_matrix = sheets[zone].iloc[0:-1, 1:].to_numpy(dtype=float)
                        is_valid, message = validate_ahp_matrix(zone_matrix)
                        if not is_valid:
                            st.error(f"Error in '{zone}' sheet: {message}")
                            raise ValueError(f"Validation failed for '{zone}' sheet.")

                

                # --- Calculate HI and Define criteria_data ---
                HI_data = calculate_hi_and_assign_zones(Prod)
                HI_data1 = HI_data.groupby('Well').last()[['Cum HI oil', 'Cum HI water']].reset_index() 
                criteria_data = HI_data1.merge(Crit, on='Well') 
                criteria_data['Zone'] = HI_data['Zone']
                
                ahp_data = {
                    'Zone 1': sheets['Zone 1'],
                    'Zone 2': sheets['Zone 2'],
                    'Zone 3': sheets['Zone 3'],
                    'Zone 4': sheets['Zone 4'],
                    'Zone 5': sheets['Zone 5']
                }
                
                # --- Plot HI Data ---
                fig, ax = plt.subplots(figsize=(15, 7))

                # Define color and marker for each well type
                well_types = {
                    '1': ('green', '^', 'High Oil & High Water'),
                    '2': ('grey', 's', 'Low Oil & Low Water'),
                    '3': ('orange', 'D', 'Low Oil & High Water'),
                    '4': ('blue', 'v', 'High Oil & Low Water'),
                    '5': ('purple', 'o', 'Origin')
                }

                for zone_id, (color, marker, label) in well_types.items():
                    well_data = HI_data[HI_data['Zone'] == zone_id]  # Group by Zone
                    ax.scatter(well_data['Cum HI oil'], well_data['Cum HI water'],
                            c=color, marker=marker, label=label, s=50)
                    for i, row in well_data.iterrows():
                        ax.text(row['Cum HI oil'], row['Cum HI water'], row['Well'], fontsize=8, ha='center', va='bottom')

                ax.set_xlabel("Cum HI oil")
                ax.set_ylabel("Cum HI water")

                # Set axis limits (adjust as needed)
                ax.set_xlim(HI_data['Cum HI oil'].min() * 1.1, HI_data['Cum HI oil'].max() * 1.1)
                ax.set_ylim(HI_data['Cum HI water'].min() * 1.1, HI_data['Cum HI water'].max() * 1.1)
                # Add horizontal and vertical lines at zero
                ax.axhline(0, color='black', linewidth=1)  # Horizontal line at y=0
                ax.axvline(0, color='black', linewidth=1)  # Vertical line at x=0

                ax.legend()
                st.pyplot(fig)

                # --- Zone Selection ---
                zone_descriptions = {
                    "1": "1 : High Oil & High Water",
                    "2": "2 : Low Oil & Low Water",
                    "3": "3 : Low Oil & High Water",
                    "4": "4 : High Oil & Low Water",
                    "5": "5 : Origin"
                }
                
                selected_zone = st.selectbox(
                    "Choose a zone:",
                    [(zone_id, description) for zone_id, description in zone_descriptions.items()],
                    format_func=lambda option: option[1]
                )
                
                # --- AHP Data Processing ---
                zone_id = selected_zone[0]
                ahp_sheet = ahp_data.get(f"Zone {zone_id}")
                
                if ahp_sheet is not None:
                    # Validate criteria types format
                    
                        # Process AHP data
                        matrice_comparaison = ahp_sheet.iloc[0:-1, 1:].to_numpy(dtype=float)
                        weights = calculer_vecteur_priorite(matrice_comparaison)
                        weights_df = pd.DataFrame(weights.reshape(1, -1), columns=ahp_sheet.columns[1:])
                        weights_df['Criteria'] = 'Weights'
                        weights_df.set_index('Criteria', inplace=True)
                        st.table(weights_df)  # Display weights table
                else:
                    st.error(f"No AHP data found for Zone {zone_id}.")
                
                # --- Perform TOPSIS and ELECTRE Analysis ---
                st.subheader(f"Analysis for Zone {zone_id}")

                # --- TOPSIS Analysis ---
                if selected_zone[0] in criteria_data['Zone'].unique():
                    zone_data = criteria_data[criteria_data['Zone'] == selected_zone[0]]
                    criteria_columns = zone_data.columns.difference(['Well', 'Zone'])
                    ahp_sheet = ahp_data.get(f"Zone {selected_zone[0]}")
                    if ahp_sheet is not None:
                        criteria_types = ahp_sheet.iloc[-1, 1:].apply(lambda x: x.lower() in ['true', 'max', 'maximize']).values
                        matrice_comparaison = ahp_sheet.iloc[0:-1, 1:].to_numpy(dtype=float)
                        weights = calculer_vecteur_priorite(matrice_comparaison)
                        data = zone_data[criteria_columns].values.astype(float)
                        weighted_normalized_data = calculate_normalized_weighted_values(data, weights)
                        ideal_point, anti_ideal_point = calculate_ideal_and_anti_ideal_points(weighted_normalized_data, criteria_types)
                        distances_to_ideal, distances_to_anti_ideal, scores = calculate_distances_and_scores(
                            weighted_normalized_data, ideal_point, anti_ideal_point
                        )
                        results_df = pd.DataFrame({
                            'Well': zone_data['Well'],
                            'Score': scores
                        }).sort_values(by='Score', ascending=True).reset_index(drop=True)
                        results_df.index += 1
                        results_df['Ranking'] = results_df.index
                        results_df = results_df[['Ranking', 'Well', 'Score']]
                        output_topsis = io.BytesIO()
                        with pd.ExcelWriter(output_topsis, engine='xlsxwriter') as writer:
                            results_df.to_excel(writer, index=False, sheet_name='Results')
                        
                        # --- ELECTRE Analysis ---
                        criteria_types = ahp_sheet.iloc[-1, 1:].apply(lambda x: x.lower() in ['true', 'max', 'maximize']).values
                        matrice_comparaison = ahp_sheet.iloc[0:-1, 1:].to_numpy(dtype=float)
                        weights = calculer_vecteur_priorite(matrice_comparaison)
                        data = zone_data[criteria_columns].values.astype(float)
                        concordance_matrix = calculate_concordance_matrix(data, weights)
                        thresholds = np.full(data.shape[1], 0.2)
                        scale_lengths = np.max(data, axis=0) - np.min(data, axis=0)
                        discordance_matrix = calculate_discordance_matrix(data, weights, thresholds, scale_lengths)
                        concordance_threshold = 0.6
                        discordance_threshold = 0.4
                        outranking_matrix = calculate_outranking_relationships(concordance_matrix, discordance_matrix, concordance_threshold, discordance_threshold)
                        ranked_actions, outranking_counts = rank_actions_electre(outranking_matrix, zone_data['Well'].values)
                        
                        ranked_df = pd.DataFrame({'Ranking': range(1, len(zone_data) + 1), 'Well': ranked_actions[::-1]})
                        ranked_df['Outranked By'] = [outranking_counts[well] for well in ranked_actions[::-1]]
                        ranked_df.reset_index(drop=True, inplace=True)
                        output_electre = io.BytesIO()
                        with pd.ExcelWriter(output_electre, engine='xlsxwriter') as writer:
                            ranked_df.to_excel(writer, index=False, sheet_name='Results')
                        
                        # --- Display Results Side-by-Side ---
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.header("TOPSIS Results")
                            st.write(results_df.reset_index(drop=True))
                            st.download_button(
                                label="Download TOPSIS Results",
                                data=output_topsis.getvalue(),
                                file_name="results_topsis.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                        with col2:
                            st.header("ELECTRE Results")
                            st.write(ranked_df)
                            st.download_button(
                                label="Download ELECTRE Results",
                                data=output_electre.getvalue(),
                                file_name="results_electre.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                    else:
                        st.error(f"No AHP data found for Zone {selected_zone[0]}. Cannot perform calculations.")
                else:
                    st.error("Invalid zone selection.")
            except ValueError as e:
                st.error(str(e))
            
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.info("Please upload an Excel file to start the analysis.")