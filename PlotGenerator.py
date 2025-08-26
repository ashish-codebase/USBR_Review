# from turtle import st
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.dates as mdates
# import pandas as pd

# class PlotGenerator:

#     def __init__(self, df, PLOT_SIZE, selected_site, date_range):
#         self.df = df
#         self.PLOT_SIZE = PLOT_SIZE
#         self.selected_site = selected_site
#         self.date_range = date_range
#         self.selected_site_bold =  f"$\\mathbf{{{selected_site}}}$"


#     def nan_check(self, col_name):
#         """Check if a column is all NaNs."""
#         if col_name not in self.df.columns:
#             return True

#         if self.df[col_name].isnull().all():
#             return True
#         else:
#             return False

#     def plot_et(self):

#         global date_range
#         """Display bar plot of daily values (from hourly summed values.)"""

#         if self.nan_check("ET"):
#             return

#         ET_Daily = self.df["ET"].resample("D").sum()

#         ET_Daily = ET_Daily / 2
#         days = ET_Daily.shape[0]
#         ticks = np.clip(int(days / 10), 1, 15)


#         fig, ax = plt.subplots(figsize=self.PLOT_SIZE)

#         bar_width = np.clip((0.0023 * ET_Daily.shape[0] + 0.0204), 0.025, 0.2)

#         ax.bar(

#             ET_Daily.index,

#             ET_Daily.values,

#             label="Cumulative daily ET (mm)",

#             width=bar_width,
#         )


#         ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))

#         ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))

#         plt.xticks(rotation=45, ha="right")

#         ax.set_xlim(self.date_range)

#         ax.set_title(self.selected_site_bold + ": Daily cumulative ET (mm).")

#         ax.legend(loc="lower left")

#         st.pyplot(fig)
#         plt.close()

