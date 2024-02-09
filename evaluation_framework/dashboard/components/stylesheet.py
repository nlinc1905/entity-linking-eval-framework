
def style_confusion_matrix(category_to_color_map: dict):
    styles = {
        'tp': {
            'color': category_to_color_map['TP'],
            'font-weight': 'bold',
            'padding': '25px',
            'border-left': '2px solid gray',
        },
        'fp': {
            'color': category_to_color_map['FP'],
            'font-weight': 'bold',
            'padding': '25px',
            'border-left': '2px solid gray',
        },
        'tn': {
            'color': category_to_color_map['TN'],
            'font-weight': 'bold',
            'padding': '25px',
        },
        'fn': {
            'color': category_to_color_map['FN'],
            'font-weight': 'bold',
            'padding': '25px',
        },
    }

    return styles


def style_graph(category_to_color_map: dict):
    stylesheet = [
        # Group selectors
        {
            'selector': 'node',
            'style': {
                'content': 'data(label)',
                'color': '#D3D3D3'
            }
        },
        {
            'selector': 'label',
            'style': {
                'content': 'data(label)',
                'color': '#708090'
            }
        },

        # Class selectors
        {
            'selector': '.tp',
            'style': {
                'line-color': category_to_color_map['TP'],
            }
        },
        {
            'selector': '.fp',
            'style': {
                'line-color': category_to_color_map['FP'],
            }
        },
        {
            'selector': '.tn',
            'style': {
                'line-color': category_to_color_map['TN'],
            }
        },
        {
            'selector': '.fn',
            'style': {
                'line-color': category_to_color_map['FN'],
            }
        },

        # Other selectors
        {
            'selector': ':selected',
            'css': {
                'background-color': 'SteelBlue',
                'line-color': 'SteelBlue',
            }
        },
    ]

    return stylesheet


def style_rank_data_table(category_to_color_map: dict):
    conditional_styles = [
        {
            'if': {
                'filter_query': '{model_score_category} = TP',
            },
            'backgroundColor': category_to_color_map['TP'],
            'color': 'white',
        },
        {
            'if': {
                'filter_query': '{model_score_category} = FP',
            },
            'backgroundColor': category_to_color_map['FP'],
            'color': 'white',
        },
        {
            'if': {
                'filter_query': '{model_score_category} = TN',
            },
            # 'backgroundColor': category_to_color_map['TN'],
            'color': category_to_color_map['TN'],
        },
        {
            'if': {
                'filter_query': '{model_score_category} = FN',
            },
            'backgroundColor': category_to_color_map['FN'],
            'color': 'white',
        },
    ]

    return conditional_styles
