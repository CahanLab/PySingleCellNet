from palettable.tableau import Tableau_20


# Annotation constants

# Colormaps

# Color list for SCN categories (using Tableau_20 palette)
tmp_color_list = Tableau_20.mpl_colors

SCN_CATEGORY_NAMES = ["Rand", "None", "Hybrid", "Intermediate", "Singular"]
indices = [14, 15, 3, 1, 0]

selected_colors = [tmp_color_list[i] for i in indices]
SCN_CATEGORY_COLOR_DICT = dict(zip(SCN_CATEGORY_NAMES, selected_colors))
del(tmp_color_list)
del(selected_colors)

# fonts

# ...

# Arbitrary strings

SCN_DIFFEXP_KEY = "scnDiffExp"