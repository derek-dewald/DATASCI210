

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import textwrap

from DATASCI210_Functions import column_statitical_review


# Save Markers and Line Styles for reference to visualization in Matplotlib
markers = [
    '.',  # Point
    ',',  # Pixel
    'o',  # Circle
    'v',  # Triangle down
    '^',  # Triangle up
    '<',  # Triangle left
    '>',  # Triangle right
    '1',  # Tri down
    '2',  # Tri up
    '3',  # Tri left
    '4',  # Tri right
    's',  # Square
    'p',  # Pentagon
    '*',  # Star
    'h',  # Hexagon2
    'H',  # Hexagon1
    '+',  # Plus
    'x',  # Cross
    'D',  # Diamond
    'd',  # Thin diamond
    '|',  # Vertical line
    '_',  # Horizontal line
]

line_styles = [
    '-',   # Solid line
    '--',  # Dashed line
    '-.',  # Dash-dot line
    ':',   # Dotted line
]

def SimpleSinglePlot(df, 
                     x_axis,
                     y_axis,
                     label,
                     title="",
                     x_axis_title="",
                     y_axis_title="Total Observations",
                     data_type='df',
                     limit_x_axis_text=0,
                     color='blue',
                     ax=None):
    '''
    Purpose:
        Simple Function to Create a Single Plot in Matplotlib.

    Parameters:
        df(df): Dataframe
        x_axis(str, or list of values): name of column which will appear on X_axis
        y_axis(str, or list of values): name of column which will appear on Y_axis
        label(str): Name of Label which will appear in Legend. 
        Title: Tile of Graph
        x_axis_title: Title of X Axis 
        y_axis_title: Title of Y Axis
        data_type: Option to allow use of Value List, opposed to Dataframe.
        ax: If you wish to place on a subplot, can reference the location on the subplot to be placed.
    
    '''

    if not x_axis_title:
        x_axis_title = x_axis
    
    if not title:
        title = f"Plot of {label} over time"

    if data_type == 'df':
        x = df[x_axis]
        y = df[y_axis]
    else:
        x = x_axis
        y = y_axis

    if ax is None:
        plt.plot(x, y, label=label,color=color)
        plt.title(title)
        plt.xlabel(x_axis_title)
        plt.ylabel(y_axis_title)
        plt.legend()
        if limit_x_axis_text:
            plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=limit_x_axis_text))
        plt.show()

    else:
        ax.plot(x, y, label=label,color=color)
        ax.set_title(title)
        ax.set_xlabel(x_axis_title)
        ax.set_ylabel(y_axis_title)
        if limit_x_axis_text:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=limit_x_axis_text))
        ax.legend()


def SimpleBarMultiple(df,
                      label_list,
                      color_list=[],
                      columns=None,
                      title="",
                      xlabel="",
                      ylabel="",
                      figsize=(12, 8),
                      x_tick_rotation=0,
                      ax=None):
    '''
    Takes a DataFrame and creates a bar graph comparing statistics for each population.
    
    Parameters:
        df (pd.DataFrame): DataFrame with statistical values for each population.
        columns (list, optional): List of columns to include in the plot. If None, all columns will be used.
        label_list (list, optional): List of labels for each population.
        title (str, optional): Title of the plot.
        figsize (tuple, optional): Size of the figure.
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib Axes object for subplots.
    '''
    if columns is None:
        columns = df.columns.tolist()
    
    if label_list is None:
        label_list = df.index.tolist()
    
    num_populations = len(df.index)
    num_stats = len(columns)

    # Check if ax is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    if not color_list:
        color_list = plt.cm.tab20.colors
    
    bar_width = 0.8 / num_populations
    index = np.arange(num_stats)
    
    for i, (population, color) in enumerate(zip(df.index, color_list)):
        bars = ax.bar(index + i * bar_width, df.loc[population, columns], bar_width, label=label_list[i], color=color)
            
    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(index + bar_width * (num_populations - 1) / 2)
    ax.set_xticklabels(columns, rotation=x_tick_rotation)
        
    ax.legend()
    
    if ax is None:
        plt.tight_layout()
        plt.show()

def column_statitical_review(df,
                             column_name,
                             segments=10,
                             exclude_blanks_from_segments=1,
                             exclude_zeroes_from_segments=1):
   
    '''
    Purpose: Generate Data Validation information Related to a individual column which are INTs, FLOATs, etc..
 
    Input: df: Any Dataframe, column_name:The Name of the Column to Review
       
    Default:
 
    Notes: Returns a dataframe of Individual Columns for variables which are Numbers
   
    '''
   
    temp_dict = {}
    temp_dict['TOTAL_RECORDS'] = len(df)
    temp_dict['TOTAL_SUM'] = df[column_name].sum()
    temp_dict['MEAN'] = df[column_name].mean()
    temp_dict['STD_DEV'] =  df[column_name].std()
    temp_dict['MEDIAN'] = df[column_name].median()
    temp_dict['MAX'] = df[column_name].max()
    temp_dict['MIN'] = df[column_name].min()
    temp_dict['ZERO_RECORDS'] = len(df[df[column_name]==0])
    temp_dict['NON_ZERO_RECORDS'] = len(df[df[column_name]!=0])
    temp_dict['NA_RECORDS'] = len(df[df[column_name].isna()])
    temp_dict['NULL_RECORDS'] = len(df[df[column_name].isnull()])
                            
    temp_df = pd.DataFrame(temp_dict.values(),index=temp_dict.keys(),columns=[column_name])
   
    if segments ==0:
        return temp_df
   
    else:
        try:
            segment_df = create_segments_from_dataframe_column(df=df,
                                                               column_name=column_name,
                                                               segments=segments,
                                                               exclude_blanks=exclude_blanks_from_segments,
                                                               exclude_zeros=exclude_zeroes_from_segments,
                                                               output_item='value_df',
                                                               segment_column_name='Segment')
            return pd.concat([temp_df,segment_df.T])
 
        except:
            print(f'Error Generating Segment Data for {column_name}')
            return temp_df 


def ColumnStatisticalReview(df,
                            column,
                            target='target'):
    '''
    Purpose: Iteratie through a Dataset, taking statistical values of columns included for review. Important remove string values.
    This functions extended beyond .describe, in that it breaks a dataframe up based on Target variable to understand how distribution differs 
    across different subsets.

    Utilizises the function column_statistical_review, which looks at a single df.

    '''


    
    all = df.copy()
    target_df = df[df[target]==1].copy()
    non_target_df = df[df[target]==0].copy()

    sr = column_statitical_review(all,column)
    sr1 = column_statitical_review(target_df,column)
    sr2 = column_statitical_review(non_target_df,column)
    final =  sr.rename(columns={column:"Population"}).merge(sr1.rename(columns={column:"Arbitrage Present"}),left_index=True,right_index=True).merge(sr2.rename(columns={column:"Arbitrage Not Present"}),left_index=True,right_index=True).T
    final['COLUMN_NAME'] = column
    return final

def CorrelationPlot(df, 
                    variable,
                    title="Correlation Heatmap", 
                    annot=True, 
                    cmap="coolwarm", 
                    figsize=(12, 8), 
                    save_path=None, 
                    ax=None):
    """


    """

    temp = pd.concat([df[df[variable].notnull()].sort_values(variable).head(10)[[variable]],
                      df[df[variable].notnull()].sort_values(variable).tail(10)[[variable]]])
        
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(temp, annot=annot, cmap=cmap, ax=ax, vmin=-1, vmax=1, center=0, linewidths=0.5, linecolor='gray', cbar_kws={"shrink": 0.75})
    ax.set_title(title, fontsize=16)

    if ax is None:
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close(fig)  # Close the figure to free memory
        else:
            plt.show()


def CorrelationPlot(df, 
                    variable,
                    title="Correlation Heatmap", 
                    annot=True, 
                    cmap="coolwarm", 
                    figsize=(12, 8), 
                    save_path=None, 
                    ax=None):
    """


    """

    temp = pd.concat([df[df[variable].notnull()].sort_values(variable).head(10)[[variable]],
                      df[df[variable].notnull()].sort_values(variable).tail(10)[[variable]]])
        
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(temp, annot=annot, cmap=cmap, ax=ax, vmin=-1, vmax=1, center=0, linewidths=0.5, linecolor='gray', cbar_kws={"shrink": 0.75})
    ax.set_title(title, fontsize=16)

    if ax is None:
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close(fig)  # Close the figure to free memory
        else:
            plt.show()


def ColumnPlotGenerator(df,
                        variable,
                        time,
                        text,
                        target,
                        export_file,
                        color_list=[],
                        summary_text="",
                        corr_df="",
                        stat_review_df="",
                        limit_x_axis_text=0):
    
    fig, axs = plt.subplots(4, 3, figsize=(20,20))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.5],hspace=0.5)
    fig.delaxes(axs[3, 0])
    fig.delaxes(axs[3, 1])

    df.sort_values('timestamp',inplace=True)

    target_df = df[df[target]==1].copy()
    not_target_df = df[df[target]==0].copy()

    SimpleSingleHist(df=df,
                     x_axis=variable,
                     label=text,
                     x_axis_title=f"{text} Population",
                     ax=axs[0,0])
    
    SimpleSingleHist(df=target_df,
                     x_axis=variable,
                     label=text,
                     x_axis_title=f"{text} Arbitrage Present",
                     color='grey',
                     ax=axs[0,1])
    
    SimpleSingleHist(df=not_target_df,
                     x_axis=variable,
                     label=text,
                     color='black',
                     x_axis_title=f"{text} Arbitrage Not Present",
                     ax=axs[0,2])

    SimpleSinglePlot(df=df.drop_duplicates(time,keep='last'),
                     x_axis=time, 
                     y_axis=variable, 
                     limit_x_axis_text=limit_x_axis_text,
                     ax=axs[1,0],
                     label=text)

    SimpleSinglePlot(df=target_df.drop_duplicates(time,keep='last'),
                     x_axis=time, 
                     y_axis=variable, 
                     limit_x_axis_text=limit_x_axis_text,
                     ax=axs[1,1],
                     color='grey',
                     label=text)

    SimpleSinglePlot(df=not_target_df.drop_duplicates(time,keep='last'),
                     x_axis=time, 
                     y_axis=variable, 
                     limit_x_axis_text=limit_x_axis_text,
                     ax=axs[1,2],
                     color='black',
                     label=text)

    if len(stat_review_df)>0:
        cols = ['Segment 1', 'Segment 2', 'Segment 3', 'Segment 4', 'Segment 5','Segment 6', 'Segment 7', 'Segment 8', 'Segment 9', 'Segment 10']
        
        sr_sgm_lists = [stat_review_df[cols].loc['Population'].tolist(),
                        stat_review_df[cols].loc['Arbitrage Present'].to_list(),
                        stat_review_df[cols].loc['Arbitrage Not Present'].tolist()]
        
        x = [x for x in range(10)]
    
        SimplePlotMultiple(df="",
                           title=f"Segment Analysis for {text}",
                           x_axis=x,
                           y_axis_list=sr_sgm_lists,
                           label_list=['Population','Arbitrage Present',"Arbitrage Not Present"],
                           marker_list=['*','o','|'],
                           linestyle_list=[':','--','-.'],
                           color_list=color_list,
                           data_type='',
                           ax=axs[2,0])

        SimpleBarMultiple(stat_review_df,
                          color_list=color_list,
                          label_list = ['Population',"Arbitrage Present",'Arbitrage Not Present'],
                          columns=['MEAN','STD_DEV','MEDIAN','MAX','MIN'],
                          title= 'Comparison of Statistics',
                          ax=axs[2,1])

        SimpleBarMultiple(stat_review_df,
                          color_list=color_list,
                          label_list = ['Population',"Arbitrage Present",'Arbitrage Not Present'],
                          columns=['TOTAL_RECORDS','NON_ZERO_RECORDS','NA_RECORDS','NULL_RECORDS'],
                          x_tick_rotation=45,
                          title='Comparison of Data Completion',
                          ax=axs[2,2])

    if len(corr_df)>0:
        CorrelationPlot(corr_df,variable,ax=axs[3,2])

    if summary_text:
        text0 = "\n".join(textwrap.wrap(summary_text[0], width=110))
        text1 = "\n".join(textwrap.wrap(summary_text[1], width=110))
        text2 = "\n".join(textwrap.wrap(summary_text[2], width=110))
        
        fig.text(0.05, 0.20, text0, ha='left', va='top', 
                 fontsize=14, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        fig.text(0.05, 0.17, text1, ha='left', va='top', 
                 fontsize=14, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        fig.text(0.05, 0.08, text2, ha='left', va='top', 
                 fontsize=14, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    if export_file:
        plt.tight_layout()
        plt.savefig(export_file)  # Save the figure to a file
        plt.close(fig)  # Close the figure to free memory
    else:
        plt.tight_layout()
        plt.show()


def SimplePlotMultiple(df,
                       x_axis,
                       y_axis_list,
                       label_list,
                       color_list,
                       title="",
                       x_axis_title="Segment",
                       y_axis_title="Segment Threshold Value",
                       x_axis_rotation=0,
                       y_axis_max_min=[],
                       data_type='df',
                       limit_x_axis_text=0,
                       marker_list="",
                       linestyle_list="",
                       ax=None):
    
    if not x_axis_title:
        x_axis_title = x_axis
    
    if data_type == 'df':
        x = df[x_axis]
        y = [df[y_axis] for y_axis in y_axis_list]
    else:
        x = x_axis
        y = y_axis_list

    if not marker_list:
        marker_list = [np.random.choice(markers) for marker in range(len(y_axis_list))]
    
    if not linestyle_list:
        linestyle_list = [np.random.choice(line_styles) for marker in range(len(y_axis_list))]

    if ax is None:
        for count,line in enumerate(y):
            plt.plot(x, line, label=label_list[count],marker=marker_list[count],linestyle=linestyle_list[count],color=color_list[count])
                
        plt.title(title)
        plt.xlabel(x_axis_title)
        plt.ylabel(y_axis_title)
        plt.legend()
        if limit_x_axis_text:
            plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=limit_x_axis_text))
        if len(y_axis_max_min)!=0:
            plt.ylim(y_axis_max_min[0], y_axis_max_min[1])
        plt.show()
    else:
        for count,line in enumerate(y):
            ax.plot(x, line, label=label_list[count],marker=marker_list[count],linestyle=linestyle_list[count],color=color_list[count])
        ax.set_title(title)
        ax.set_xlabel(x_axis_title)
        ax.set_ylabel(y_axis_title)
        if x_axis_rotation!=0:
            plt.xticks(rotation=x_axis_rotation)
        if limit_x_axis_text:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=limit_x_axis_text))
        if len(y_axis_max_min)!=0:
            ax.set_ylim(y_axis_max_min[0], y_axis_max_min[1])
        ax.legend()


def SimpleSingleHist(df, 
                     x_axis,
                     label,
                     title="",
                     x_axis_title="",
                     y_axis_title="Total Observations",
                     data_type='df',
                     color='blue',
                     y_axis_max_min=[],
                     ax=None):
    
    if not x_axis_title:
        x_axis_title = x_axis
    if not title:
        title = f"Histogram of {label}"

    if data_type == 'df':
        x = df[x_axis]
    else:
        x = x_axis

    if ax is None:
        plt.hist(x, label=label,color=color)
        plt.title(title)
        plt.xlabel(x_axis_title)
        plt.ylabel(y_axis_title)
        plt.legend()
        plt.show()
    else:
        ax.hist(x, label=label,color=color)
        ax.set_title(title)
        ax.set_xlabel(x_axis_title)
        ax.set_ylabel(y_axis_title)
        if len(y_axis_max_min)!=0:
            ax.set_ylim(y_axis_max_min[0], y_axis_max_min[1])
        ax.legend()


def Heatmap_From_Df(df, 
                    index,
                    column,
                    value,
                    aggfunc,
                    index_sort="",
                    column_sort="",
                    title="Correlation Heatmap", 
                    annot=False, 
                    cmap="viridis",
                    x_axis_title="",
                    y_axis_title='',
                    annot_fontsize=8,
                    figsize=(12, 8), 
                    save_path=None, 
                    ax=None):
    """
    Creates a correlation heatmap from a DataFrame.
    
    Parameters:

    """

    df1 = df.pivot_table(index=index,columns=column,values=value,aggfunc=aggfunc).fillna(0)

    
    if len(column_sort)!=0:
        c_order = df.drop_duplicates(column).sort_values(column_sort)[column].tolist()
        df1 = df1[c_order]
    
    if len(index_sort)!=0:
        i_order = df.drop_duplicates(index).sort_values(index_sort)[index].tolist()
        df1 = df1.reindex(i_order)
     
    # Create a figure if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Create the heatmap
    sns.heatmap(df1,cmap=cmap,annot=annot,annot_kws={"size": annot_fontsize}, ax=ax)
    
    # Set title
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_axis_title, fontsize=14)
    ax.set_ylabel(y_axis_title, fontsize=14)
    
    
    # Adjust layout to make space for the color bar and titles
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to free memory
    if ax is None:
        plt.show()

