from palettable.cartocolors.qualitative import Safe_6
from palettable.tableau import GreenOrange_6

# Annotation constants

# Colormaps

tmp_color_list = GreenOrange_6.mpl_colors
SCN_CATEGORY_NAMES = ["Singular", "None", "Parent.Child", "Sibling", "Hybrid", "Gp.Gc"]
SCN_CATEGORY_COLOR_DICT = dict(zip(SCN_CATEGORY_NAMES, tmp_color_list))
del(tmp_color_list)


# fonts

# ...