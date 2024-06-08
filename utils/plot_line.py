
# Interactive Plot for the voltage and current
# bokeh docs: https://docs.bokeh.org/en/2.4.1/docs/first_steps/first_steps_1.html
import bokeh.plotting as bplt
from bokeh.io import curdoc
from bokeh.models import BoxAnnotation
from math import ceil

# Apply the theme to the plot
curdoc().theme = "caliber"  # Can be one of "caliber", "dark_minimal", "light_minimal", "night_sky", "contrast"

# colors list
line_colors = ['#23943b', '#1f77b4']  # TODO: Add more colors if needed

# --------------------------------------------------------- #
# ---------- Map the annotated label to a color ----------- #
# --------------------------------------------------------- #
color_map = {                  
    'Spike': 'red',
    'Fast-Ripple': 'blue',
    'Ripple': 'green',  
    'Spike+Ripple': 'yellow',
    'Spike+Fast-Ripple': 'pink',
    'Ripple+Fast-Ripple': 'cyan',
    'Spike+Ripple+Fast-Ripple': 'black'
}

""" Only possible in python>= 3.12 
type BoxAnnotationParams = {
    "bottom": float,
    "top": float,
    "left": float,
    "right": float,
    "fill_alpha": float,
    "fill_color": str
} """

"""
create_fig: Create a figure with the given parameters

Args:
    title (str): Title of the plot
    x_axis_label (str): Label of the x-axis
    y_axis_label (str): Label of the y-axis
    x (np.ndarray): Array of x values
    y_arrays (list): List of tuples containing the y values and the legend label
    x_range (tuple): Range of the x-axis    (Can make linked ranges with other plots)
    y_range (tuple): Range of the y-axis
    sizing_mode (str): Sizing mode of the plot
    tools (str): Tools to be added to the plot
    tooltips (str): Tooltips to be added to the plot
    legend_location (str): Location of the legend
    legend_bg_fill_color (str): Background fill color of the legend
    legend_bg_fill_alpha (float): Background fill alpha of the legend
    box_annotation_params (dict): Parameters to create a box annotation
Returns:
    bplt.Figure: The plot
"""
def create_fig(title, x_axis_label, y_axis_label, 
               x, y_arrays, x_range=None, y_range=None,
               sizing_mode=None, tools=None, tooltips=None, 
               legend_location=None, legend_bg_fill_color=None, legend_bg_fill_alpha=None, 
               box_annotation_params=None):
    # Create the plot
    p = bplt.figure(
        title=title,
        x_axis_label=x_axis_label, 
        y_axis_label=y_axis_label,
        sizing_mode=sizing_mode or "stretch_both",    # Make the plot stretch in both width and height
        tools=tools or "pan, box_zoom, wheel_zoom, hover, undo, redo, zoom_in, zoom_out, reset, save",
        tooltips=tooltips or "Data point @x: @y"
    )

    # Set the range of the x and y-axis
    if x_range is not None:
        p.x_range = x_range
    if y_range is not None:
        p.y_range = y_range

    # Add a line graph to the plot for each y_array
    for (arr_idx, y_array) in enumerate(y_arrays):
        p.line(x, y_array[0], legend_label=y_array[1], line_width=1, line_color=line_colors[arr_idx % len(line_colors)])

    # Legend settings
    p.legend.location = legend_location or "top_right"
    p.legend.background_fill_color = legend_bg_fill_color or "navy"
    p.legend.background_fill_alpha = legend_bg_fill_alpha or 0.1
    p.legend.click_policy = "hide"  # Clicking on a legend item will hide the corresponding line
    # Format legend to 2 columns
    p.legend.ncols = ceil(len(y_arrays) / 7)    # Make the number of rows no more than 7

    # Grid settings
    # p.ygrid.grid_line_color = "red"

    # Add a box annotation
    if box_annotation_params is not None:
        inner_box = BoxAnnotation(
            bottom=box_annotation_params["bottom"], 
            top=box_annotation_params["top"], 
            left=box_annotation_params["left"], 
            right= box_annotation_params["right"], 
            fill_alpha=box_annotation_params["fill_alpha"], 
            fill_color=box_annotation_params["fill_color"],
        )

        p.add_layout(inner_box)

    # Change the number of decimal places on hover
    p.hover.formatters = {'@x': 'numeral', '@y': 'numeral'}
    p.hover.tooltips = [("x", "@x{0.0}"), ("y", "@y{0.0000}")]

    # Return the plot
    return p
