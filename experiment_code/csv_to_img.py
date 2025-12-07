import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your Data
file_path = "../results_to_use/weight_tuning_results_p2.csv"  # Your file name
df = pd.read_csv(file_path)

# --- THE FIX: FILTER THE DATA ---
# Sort by Win Rate (Highest first)
df_sorted = df.sort_values(by=['winrate', 'wins'], ascending=False)

# Take the Top 5 (The Winners)
top_rows = df_sorted.head(5)

# Take the Bottom 2 (The Losers - for contrast)
# bottom_rows = df_sorted.tail(2)

# Combine them into one small, readable table
df_display = pd.concat([top_rows])

# 2. Setup the plot (Standard size is fine now because we have fewer rows)
fig, ax = plt.subplots(figsize=(8, 4)) 
ax.axis('tight')
ax.axis('off')

# 3. Create the table
table = ax.table(cellText=df_display.values,
                 colLabels=df_display.columns,
                 cellLoc='center',
                 loc='center')

# 4. Styling (Make it readable for a slide)
table.auto_set_font_size(False)
table.set_fontsize(14) # Big font for projectors
table.scale(1.2, 2.5)  # Make rows tall and readable

# 5. Advanced Styling: Highlight the Winners vs Losers
for (row, col), cell in table.get_celld().items():
    if row == 0:
        # Header
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#40466e')
    elif row <= 5: 
        # The Top 5 Rows (Indices 1-5 in table) - Light Blue
        cell.set_facecolor('#e6f2ff')
    else:
        # The Bottom Rows - Light Red (to show they are worse)
        cell.set_facecolor('#ffe6e6')

# 6. Save
plt.title("Phase 2: Tuning Highlights", fontweight="bold", y=1.05)
plt.savefig("tuning_highlights_table.png", bbox_inches='tight', dpi=300)
plt.show()