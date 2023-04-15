def nice_table(dict, title=''):
    html = f'<h2 style="text-align:left;">{title}</h2>'
    html += '<table style="width:50%; border-collapse: collapse; font-size: 16px; text-align:left; padding: 10px; border: 1px solid #fff;">'
    color_index = 0
    for key, value in dict.items():
        row_color = 'white'
        html += f'<tr><td style="border: 1px solid #fff; text-align:left; padding: 10px; color: {row_color}; border-right: 1px solid #fff;">{key}</td><td style="border: 1px solid #fff; text-align:left; padding: 10px; color: {row_color}; opacity: 0.8; border-left: 1px solid #fff;">{value}</td></tr>'
        color_index = (color_index + 1) % 6
    html += '</table>'
    return html

