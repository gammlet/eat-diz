import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
# Dict with entrys
entries = {}
mat_entries = {}

def show_message():
    # Get all input values
    inputs = [
        entries["ent1"].get(), entries["ent2"].get(), entries["ent3"].get(), entries["ent4"].get(), entries["ent5"].get(),
        entries["ent6"].get(), entries["ent7"].get(), entries["ent8"].get(), entries["ent9"].get(), entries["ent10"].get(),
        mat_entries["mat1"].get(), mat_entries["mat2"].get(), mat_entries["mat3"].get()
    ]

    # Check for empty entries
    if val.get() == 0:
        inputs.append(A_ent.get())
    empty_fields = [f"Field {i + 1}" for i, val in enumerate(inputs) if not val]
    if empty_fields:
        messagebox.showerror("Error", f"Please complete these entries:\n{', '.join(empty_fields)}")
        return

    # === Physical Constants ===
    R = 8.31  # Gas constant (J/mol·K)

    # === Material Paramat_entries["mat1"].get()
    Delta_h_star = int(mat_entries["mat1"].get())  # Activation enthalpy (J/mol)
    x = float(mat_entries["mat2"].get())  # Mixing parameter
    beta = float(mat_entries["mat3"].get())  # Stretching exponent (KWW exponent)

    # Calculate pre-exponential factor A
    
    
    #################### A #####################

    if val.get() == 1:
        Tg = 500  # Glass transition temperature (K)
        tau_Tg = 100  # Relaxation time at Tg (seconds) 
        A = tau_Tg / np.exp(Delta_h_star / (R * Tg))
        print(f"Calculated A = {A:.3e}")
    else:
        A = float(A_ent.get())

    # === Simulation Parameters ===
    dt = 1.0  # Time step (seconds)

    # === Create Temperature Profile ===
    cooling_rate = float( entries["ent1"].get())  # K/s
    T_start_cool = int(entries["ent2"].get())
    T_end_cool = int(entries["ent3"].get())
    cooling_time = (T_start_cool - T_end_cool) / cooling_rate
    steps_cool = int(cooling_time / dt)
    T_cool = np.linspace(T_start_cool, T_end_cool, steps_cool)

    heating_rate = float(entries["ent4"].get())  # K/s
    T_start_heat = int(entries["ent3"].get())
    T_end_heat = int(entries["ent5"].get())
    heating_time = (T_end_heat - T_start_heat) / heating_rate
    steps_heat = int(heating_time / dt)
    T_heat = np.linspace(T_start_heat, T_end_heat, steps_heat)

    hold_time = float(entries["ent6"].get()) * 3600  # hours to seconds
    steps_iso = int(hold_time / dt)
    T_iso = np.full(steps_iso, T_end_heat)

    cooling_rate2 = float(entries["ent7"].get())  # K/s
    T_start_cool2 = int(entries["ent5"].get())
    T_end_cool2 = int(entries["ent8"].get())
    cooling_time2 = (T_start_cool2 - T_end_cool2) / cooling_rate2
    steps_cool2 = int(cooling_time2 / dt)
    T_cool2 = np.linspace(T_start_cool2, T_end_cool2, steps_cool2)

    heating_rate2 = float(entries["ent9"].get())  # K/s
    T_start_heat2 = int(entries["ent8"].get())
    T_end_heat2 = int(entries["ent10"].get())
    heating_time2 = (T_end_heat2 - T_start_heat2) / heating_rate2
    steps_heat2 = int(heating_time2 / dt)
    T_heat2 = np.linspace(T_start_heat2, T_end_heat2, steps_heat2)

##################################################################SEPARATE
    # Combine all segments
    T_profile = np.concatenate([T_cool, T_heat, T_iso, T_cool2, T_heat2])
    total_steps = len(T_profile)
    time_hours = np.arange(total_steps) * dt / 3600  # Convert to hours

    # Relaxation Time Function
    def tau(T, Tf):
        return A * np.exp(
            (x * Delta_h_star) / (R * T) +
            ((1 - x) * Delta_h_star) / (R * Tf)
        )

    # Initialize Fictive Temperature
    Tf = np.zeros(total_steps)
    Tf[0] = T_profile[0]

    # Time-stepping loop
    for i in range(1, total_steps):
        T_prev = T_profile[i - 1]
        Tf_prev = Tf[i - 1]
        T_now = T_profile[i]

        tau_prev = tau(T_prev, Tf_prev)
        reduced_time = dt / tau_prev

        if tau_prev > 0:
            relaxation_factor = 1 - np.exp(-(reduced_time ** beta))
        else:
            relaxation_factor = 1

        Tf[i] = Tf_prev + (T_now - Tf_prev) * relaxation_factor

    # === Plot Results ===
    plt.figure(figsize=(8, 8))

    # === Period Boundaries ===
    cool_end = len(T_cool)
    heat_start = cool_end
    heat_end = heat_start + len(T_heat)
    iso_start = heat_end
    iso_end = iso_start + len(T_iso)
    cool2_start = iso_end
    cool2_end = cool2_start + len(T_cool2)
    heat2_start = cool2_end
    heat2_end = heat2_start + len(T_heat2)

    # === Colors for periods ===
    cool_color = '#FF0000'  # 
    heat_color = '#00FF00'  # Orae
    iso_color = '#2ca02c'   # Green
    cool2_color = '#4a3a32'  # Night
    heat2_color = '#A849C9'  # yl

    # Temperature profile
    plt.subplot(2, 1, 1)
    plt.plot(time_hours, T_profile, 'b-', label='Actual Temperature')
    plt.plot(time_hours, Tf, 'r--', label='Fictive Temperature')
    plt.ylabel('Temperature (K)')

    # Add vertical lines for segment boundaries
    segment_ends = np.cumsum([len(T_cool), len(T_heat), len(T_iso), len(T_cool2)]) * dt / 3600
    for end in segment_ends:
        plt.axvline(end, color='k', linestyle='--', alpha=0.5)

    plt.grid(True)
    plt.title('Temperature Profile')
    plt.legend()
    plt.subplot(2, 1, 2)
    # Fictive temperature vs actual temperature
    plt.plot(T_profile[:cool_end], Tf[:cool_end], color=cool_color, label='Cooling')
    plt.plot(T_profile[heat_start:heat_end], Tf[heat_start:heat_end], color=heat_color, label='Heating')
    plt.plot(T_profile[iso_start:iso_end], Tf[iso_start:iso_end], color=iso_color, label='Isothermal Hold')
    plt.plot(T_profile[cool2_start:cool2_end], Tf[cool2_start:cool2_end], color=cool2_color, label='Cooling 2')
    plt.plot(T_profile[heat2_start:heat2_end], Tf[heat2_start:heat2_end], color=heat2_color, label='Heating 2')
    
    plt.plot(T_profile, T_profile, 'k--', label='Equilibrium')
    plt.xlabel('Actual Temperature (K)')
    plt.ylabel('Fictive Temperature (K)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # === Create Results Table ===
    create_results_table(time_hours, T_profile, Tf,
                         [len(T_cool), len(T_heat), len(T_iso), len(T_cool2), len(T_heat2)])


def create_results_table(time_hours, T_profile, Tf, segment_lengths):
    # Create a new window for the table
    table_window = tk.Toplevel()
    table_window.title("Simulation Results Table")
    table_window.geometry("900x600")

    # Create Treeview widget
    columns = ("Time (h)", "Temp (K)", "Fictive Temp (K)", "Difference (K)", "Segment")
    results_table = ttk.Treeview(table_window, columns=columns, show="headings", height=25)

    # Configure columns
    col_widths = [100, 100, 120, 120, 150]
    for col, width in zip(columns, col_widths):
        results_table.heading(col, text=col)
        results_table.column(col, width=width, anchor='center')

    # Add scrollbars
    scroll_y = ttk.Scrollbar(table_window, orient="vertical", command=results_table.yview)
    scroll_x = ttk.Scrollbar(table_window, orient="horizontal", command=results_table.xview)
    results_table.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

    # Pack widgets
    results_table.pack(side="top", fill="both", expand=True)
    scroll_y.pack(side="right", fill="y")
    scroll_x.pack(side="bottom", fill="x")

    # Define segments
    segment_names = ["Cooling 1", "Heating 1", "Isothermal", "Cooling 2", "Heating 2"]
    segments = []
    start = 0
    for length, name in zip(segment_lengths, segment_names):
        end = start + length
        segments.append((start, end, name))
        start = end

    # Populate table with sample data (every 100 steps for performance)
    for i in range(0, len(time_hours), 100):
        # Find which segment this step belongs to
        segment_name = next(name for start, end, name in segments if start <= i < end)
        temp_diff = T_profile[i] - Tf[i]

        results_table.insert("", "end", values=(
            f"{time_hours[i]:.2f}",
            f"{T_profile[i]:.1f}",
            f"{Tf[i]:.1f}",
            f"{temp_diff:.1f}",
            segment_name
        ))

    # Add export button
    def export_to_csv():
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                # Write headers
                f.write(",".join(columns) + "\n")
                # Write data
                for i in range(0, len(time_hours), 100):
                    segment_name = next(name for start, end, name in segments if start <= i < end)
                    temp_diff = T_profile[i] - Tf[i]
                    line = (
                        f"{time_hours[i]:.2f},"
                        f"{T_profile[i]:.1f},"
                        f"{Tf[i]:.1f},"
                        f"{temp_diff:.1f},"
                        f"{segment_name}\n"
                    )
                    f.write(line)
            messagebox.showinfo("Export Complete", f"Data saved to {filename}")

    export_btn = ttk.Button(table_window, text="Export to CSV", command=export_to_csv)
    export_btn.pack(pady=10)


root = tk.Tk()
root.title("Glass")
root.geometry("1000x500")
# root.attributes("-fullscreen", True)

main_frame = tk.Frame(root, padx=20, pady=20)
main_frame.pack(fill=tk.BOTH, expand=True)


###############SIM FRAME####################
data1_frame = tk.LabelFrame(main_frame, text="Simulation Data", padx=10, pady=10)
data1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
fields = [
    "Simulation Data", "Enter cooling rate K/s",
    "Enter start temp for cooling K",
    "Enter start temp for heating (same as end temp for cooling) K",
    "Enter heating rate K/s",
    "Enter end heating temp(same as isotermic hold temp," + "\n" + " and start temp of second cooling period) K",
    "Enter isotermic hold time 1 or 0.5 hr"
    "Enter second cooling rate K/s",
    "Enter second cooling end temp " + "\n" + "(same as start temp for second heating) K",
    "Enter second heating rate K/s",
    "Enter second heating end temp K"
    ]


for i, field in enumerate(fields, 1):
    ttk.Label(data1_frame, text=field).grid(row=i,  column=0, sticky="w", padx=6, pady=6)

    entries[f"ent{i}"] = ttk.Entry(data1_frame)
    entries[f"ent{i}"].grid(row=i,  column=2, sticky="w", padx=6, pady=6)

############MATERIAL FRAME#################################

material_frame = tk.LabelFrame(main_frame, text="Material Data", padx=10, pady=10)
material_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
material_fields = [
    "Enter delta h star (J/mol) :",
    "Enter x Mixing parameter:",
    "Enter beta Stretching exponent:"
    ]

for i, field in enumerate(material_fields, 1):
    ttk.Label(material_frame, text=field).grid(row=i,  column=0, sticky="w", padx=6, pady=6)

    mat_entries[f"mat{i}"] = ttk.Entry(material_frame)
    mat_entries[f"mat{i}"].grid(row=i,  column=2, sticky="w", padx=6, pady=6)

################### A #########################

val  = tk.IntVar(value = 0)

A_enter_btn = ttk.Radiobutton(material_frame, text= "Enter A ", value=0, variable = val).grid(row=4, column=0, sticky="w", padx=6, pady=6)
A_enter_btn = ttk.Radiobutton(material_frame, text= "Use Calculated A ", value=1, variable = val).grid(row=5, column=0, sticky="w", padx=6, pady=6)

A_ent = ttk.Entry(material_frame)
A_ent.grid(row=4, column=1, sticky="w", padx=6, pady=6)

################### A #########################

##################TRASH#####################################
button_frame = tk.Frame(root)
button_frame.pack(pady=20)

btn = ttk.Button(button_frame, text="Run Simulation", command=show_message)
btn.grid(row=10, column=0, padx=6, pady=6)

dop_frame = tk.Frame(root)
dop_frame.pack(pady=20)

label = ttk.Label(dop_frame, text="hello there")
label.grid(row=11, column=0, padx=6, pady=6)

entries["ent1"].insert(0, 0.5)
entries["ent2"].insert(0, 500)
entries["ent3"].insert(0, 300)
entries["ent4"].insert(0, 0.5)
entries["ent5"].insert(0, 500)
entries["ent6"].insert(0, 0.5)
entries["ent7"].insert(0, 0.7)
entries["ent8"].insert(0, 300)
entries["ent9"].insert(0, 0.3)
entries["ent10"].insert(0, 500)
mat_entries["mat1"].insert(0, 315000) #delta h star
mat_entries["mat2"].insert(0, 0.5) # x
mat_entries["mat3"].insert(0, 0.7) # beta
A_ent.insert(0, 4e-35)

root.mainloop()
