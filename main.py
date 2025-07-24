import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox


def show_message():
    # Get all input values
    inputs = [
        ent1.get(), ent2.get(), ent3.get(), ent4.get(), ent5.get(),
        ent6.get(), ent7.get(), ent8.get(), ent9.get(), ent10.get(),
        delta_h_star_ent.get(), x_ent.get(), beta_ent.get()
    ]

    # Check for empty entries
    empty_fields = [f"Field {i + 1}" for i, val in enumerate(inputs) if not val]
    if empty_fields:
        messagebox.showerror("Error", f"Please complete these entries:\n{', '.join(empty_fields)}")
        return

    # === Physical Constants ===
    R = 8.31  # Gas constant (J/mol·K)

    # === Material Parameters ===
    Delta_h_star = int(delta_h_star_ent.get())  # Activation enthalpy (J/mol)
    x = float(x_ent.get())  # Mixing parameter
    beta = float(beta_ent.get())  # Stretching exponent (KWW exponent)

    # Calculate pre-exponential factor A
    Tg = 500  # Glass transition temperature (K)
    tau_Tg = 100  # Relaxation time at Tg (seconds)
    A = tau_Tg / np.exp(Delta_h_star / (R * Tg))
    print(f"Calculated A = {A:.3e}")

    # === Simulation Parameters ===
    dt = 1.0  # Time step (seconds)

    # === Create Temperature Profile ===
    cooling_rate = float(ent1.get())  # K/s
    T_start_cool = int(ent2.get())
    T_end_cool = int(ent3.get())
    cooling_time = (T_start_cool - T_end_cool) / cooling_rate
    steps_cool = int(cooling_time / dt)
    T_cool = np.linspace(T_start_cool, T_end_cool, steps_cool)

    heating_rate = float(ent4.get())  # K/s
    T_start_heat = int(ent3.get())
    T_end_heat = int(ent5.get())
    heating_time = (T_end_heat - T_start_heat) / heating_rate
    steps_heat = int(heating_time / dt)
    T_heat = np.linspace(T_start_heat, T_end_heat, steps_heat)

    hold_time = float(ent6.get()) * 3600  # hours to seconds
    steps_iso = int(hold_time / dt)
    T_iso = np.full(steps_iso, T_end_heat)

    cooling_rate2 = float(ent7.get())  # K/s
    T_start_cool2 = int(ent5.get())
    T_end_cool2 = int(ent8.get())
    cooling_time2 = (T_start_cool2 - T_end_cool2) / cooling_rate2
    steps_cool2 = int(cooling_time2 / dt)
    T_cool2 = np.linspace(T_start_cool2, T_end_cool2, steps_cool2)

    heating_rate2 = float(ent9.get())  # K/s
    T_start_heat2 = int(ent8.get())
    T_end_heat2 = int(ent10.get())
    heating_time2 = (T_end_heat2 - T_start_heat2) / heating_rate2
    steps_heat2 = int(heating_time2 / dt)
    T_heat2 = np.linspace(T_start_heat2, T_end_heat2, steps_heat2)

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

    # Fictive temperature vs actual temperature
    plt.subplot(2, 1, 2)
    plt.plot(T_profile, Tf, 'r-', label='Fictive Temperature')
    plt.plot(T_profile, T_profile, 'k--', label='Equilibrium')
    plt.xlabel('Actual Temperature (K)')
    plt.ylabel('Fictive Temperature (K)')
    plt.title('Fictive vs Actual Temperature')
    plt.grid(True)
    plt.legend()

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

# cooling rate
label1 = ttk.Label(data1_frame, text="Enter cooling rate K/s")
label1.grid(row=0, column=0, sticky="w", padx=6, pady=6)
ent1 = ttk.Entry(data1_frame)
ent1.grid(row=0, column=1, padx=6, pady=6)

# start temp C
label2 = ttk.Label(data1_frame, text="Enter start temp for cooling K")
label2.grid(row=1, column=0, sticky="w", padx=6, pady=6)
ent2 = ttk.Entry(data1_frame)
ent2.grid(row=1, column=1, padx=6, pady=6)

# start temp H = end temp C
label3 = ttk.Label(data1_frame, text="Enter start temp for heating (same as end temp for cooling) K")
label3.grid(row=2, column=0, sticky="w", padx=6, pady=6)
ent3 = ttk.Entry(data1_frame)
ent3.grid(row=2, column=1, padx=6, pady=6)

# heating rate
label4 = ttk.Label(data1_frame, text="Enter heating rate K/s")
label4.grid(row=3, column=0, sticky="w", padx=6, pady=6)
ent4 = ttk.Entry(data1_frame)
ent4.grid(row=3, column=1, padx=6, pady=6)

# hold temp
label5 = ttk.Label(data1_frame,
                   text="Enter end heating temp(same as isotermic hold temp," + "\n" + " and start temp of second cooling period) K")
label5.grid(row=4, column=0, sticky="w", padx=6, pady=6)
ent5 = ttk.Entry(data1_frame)
ent5.grid(row=4, column=1, padx=6, pady=6)

# hold time
label6 = ttk.Label(data1_frame, text="Enter isotermic hold time 1 or 0.5 hr")
label6.grid(row=5, column=0, sticky="w", padx=6, pady=6)
ent6 = ttk.Entry(data1_frame)
ent6.grid(row=5, column=1, padx=6, pady=6)

# second cooling rate
cool2_rate = ttk.Label(data1_frame, text="Enter second cooling rate K/s")
cool2_rate.grid(row=6, column=0, sticky="w", padx=6, pady=6)
ent7 = ttk.Entry(data1_frame)
ent7.grid(row=6, column=1, padx=6, pady=6)

# cooling 2 end temp (same as start heating 2 temp)
cooling2_end_temp = ttk.Label(data1_frame,
                              text="Enter second cooling end temp " + "\n" + "(same as start temp for second heating) K")
cooling2_end_temp.grid(row=7, column=0, sticky="w", padx=6, pady=6)
ent8 = ttk.Entry(data1_frame)
ent8.grid(row=7, column=1, padx=6, pady=6)

# heating2 rate
heating2_rate = ttk.Label(data1_frame, text="Enter second heating rate K/s")
heating2_rate.grid(row=8, column=0, sticky="w", padx=6, pady=6)
ent9 = ttk.Entry(data1_frame)
ent9.grid(row=8, column=1, padx=6, pady=6)

# heating2 end temp
heating2_end_temp = ttk.Label(data1_frame, text="Enter second heating end temp K")
heating2_end_temp.grid(row=9, column=0, sticky="w", padx=6, pady=6)
ent10 = ttk.Entry(data1_frame)
ent10.grid(row=9, column=1, padx=6, pady=6)

############MATERIAL FRAME#################################
material_frame = tk.LabelFrame(main_frame, text="Material Data", padx=10, pady=10)
material_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

delta_h_star_label = ttk.Label(material_frame, text="Enter delta h star (J/mol) :")
delta_h_star_label.grid(row=0, column=0, sticky="w", padx=6, pady=6)
delta_h_star_ent = ttk.Entry(material_frame)
delta_h_star_ent.grid(row=0, column=1, padx=6, pady=6)

x_label = ttk.Label(material_frame, text="Enter x Mixing parameter:")
x_label.grid(row=1, column=0, sticky="w", padx=6, pady=6)
x_ent = ttk.Entry(material_frame)
x_ent.grid(row=1, column=1, padx=6, pady=6)

beta_label = ttk.Label(material_frame, text="Enter beta Stretching exponent:")
beta_label.grid(row=2, column=0, sticky="w", padx=6, pady=6)
beta_ent = ttk.Entry(material_frame)
beta_ent.grid(row=2, column=1, padx=6, pady=6)

A_label = ttk.Label(material_frame, text="Enter A(does not work):")
A_label.grid(row=3, column=0, sticky="w", padx=6, pady=6)
A_ent = ttk.Entry(material_frame)
A_ent.grid(row=3, column=1, padx=6, pady=6)

##################TRASH#####################################
button_frame = tk.Frame(root)
button_frame.pack(pady=20)

btn = ttk.Button(button_frame, text="Run Simulation", command=show_message)
btn.grid(row=10, column=0, padx=6, pady=6)

dop_frame = tk.Frame(root)
dop_frame.pack(pady=20)

label = ttk.Label(dop_frame, text="hello there")
label.grid(row=11, column=0, padx=6, pady=6)

ent1.insert(0, 0.5)
ent2.insert(0, 500)
ent3.insert(0, 300)
ent4.insert(0, 0.5)
ent5.insert(0, 500)
ent6.insert(0, 0.5)
ent7.insert(0, 0.7)
ent8.insert(0, 300)
ent9.insert(0, 0.3)
ent10.insert(0, 500)
delta_h_star_ent.insert(0, 315000)
x_ent.insert(0, 0.5)
beta_ent.insert(0, 0.7)

root.mainloop()
