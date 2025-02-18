from palettable.cartocolors.qualitative import Safe_6
# from palettable.tableau import GreenOrange_6
from palettable.tableau import GreenOrange_12
from palettable.tableau import Tableau_20


# Annotation constants

# Colormaps

# see https://jiffyclub.github.io/palettable/tableau/#greenorange_12
tmp_color_list = GreenOrange_12.mpl_colors
tmp_color_list = Tableau_20.mpl_colors

# SCN_CATEGORY_NAMES = ["Rand", "None", "Singular", "Parent.Child", "Sibling", "Hybrid"]
# SCN_CATEGORY_NAMES = ["Singular", "Intermediate", "Hybrid", "None", "Rand"]
# indices = [0, 1, 3, 15, 14]
SCN_CATEGORY_NAMES = ["Rand", "None", "Hybrid", "Intermediate", "Singular"]
indices = [14, 15, 3, 1, 0]

selected_colors = [tmp_color_list[i] for i in indices]
### SCN_CATEGORY_COLOR_DICT = dict(zip(SCN_CATEGORY_NAMES, tmp_color_list))
SCN_CATEGORY_COLOR_DICT = dict(zip(SCN_CATEGORY_NAMES, selected_colors))
del(tmp_color_list)
del(selected_colors)

# fonts

# ...

# Arbitrary strings

SCN_DIFFEXP_KEY = "scnDiffExp"