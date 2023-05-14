from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from IPython.display import Image as Im
from IPython.display import display
import imageio
from tqdm import tqdm
import os
from PIL import Image, ImageOps

class VisualizeModel():
    '''
    This class allows visualization of data and decision regions in 2D and 3D.
    '''
    def __init__(self, name, x_data, y_data, clf):
        '''
        The init method takes the dataset x_data, y_data and the classifier clf for which plots are to be made.
        x_data is assumed to be 3D. If it is not, the first three columns are taken.
        '''
        
        self.x_data = x_data.iloc[:, -3:]
        self.y_data = y_data
        self.fps = 15
        self.filename = name
        self.clf = clf

    def illustrate_features_3D(self, animated=False, show=False):
        '''
        Show a 3D plot of the feature space using the points in x_data and the labels in y_data
        '''
        x_data_r, y_data_r = self.x_data.values, self.y_data
        self.clf.fit(x_data_r, y_data_r)                                        # fit the classifier
        
        # Basic Plot Setup
        plt.style.use('dark_background')                                        # dark background           
        fig = plt.figure()
        fig.set_dpi(200)                                                        # increase resolution
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)                                                          # remove grid
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False    # remove principal planes
        ax.set_axis_off()                                                       # remove axis              
        f1, f2, f3 = self.x_data.columns                                        # get feature names
        ax.set_title(f'Plotting {f1} (x) vs {f2} (y) vs {f3} (z)', fontsize=10) # add title
        
        # Scatter Plot the Data
        colors = np.array(['#799FFA', '#ffff00', '#5fff4a', '#f781bf'])
        scatter = ax.scatter(x_data_r[:,0], x_data_r[:,1], x_data_r[:,2], c=y_data_r, cmap=matplotlib.colors.ListedColormap(colors))
        
        # Increase the vertical scale (z axis)
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 1.5, 1]))
        
        # Add a legend for the classes
        legend_elements = [(marker, label) for marker, label in zip(scatter.legend_elements()[0], ['Body Level 0', 'Body Level 1', 'Body Level 2', 'Body Level 3'])]
        ax.legend(*zip(*legend_elements), loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=6)
        
        if not animated and show:
            plt.show()                  # Show a static plot (code above is enough)
        else:
            plt.close()
            frames = []
            step = 4
            num_frames = 360//step
            for i in tqdm(range(num_frames)):
                ax.view_init(azim=i*step, elev=0)
                # generate the figure in memory without showing
                fig.canvas.draw()                     
                # convert the saved figure to a 1D numpy array                             
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)   
                # ... 3D numpy array (width, height)->(height, width)->(height, width, 3)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))  
                # append it into the list of frames that make up the gif
                frames.append(frame)   
                                                                                
            # save the list of images as an animated GIF
            path = f'../../Saved/{self.filename}.gif' 
            imageio.mimsave(path, frames,format='gif', fps=self.fps)          # make a gif out of the frames where there are 15 frames per second
            if show:    display(Im(path))

    def illustrate_features_2D(self, show=False):
        '''
        Show 2D plot with decision regions
        '''
        # Take the two top-most features
        x_data_r, y_data_r = self.x_data.iloc[:,1:].values, self.y_data
        self.clf.fit(x_data_r, y_data_r)  
        
        # Prepare the x,y grid so it spans all points
        x1_min, x1_max = x_data_r[:, 0].min() - 1, x_data_r[:, 0].max() + 1
        x2_min, x2_max = x_data_r[:, 1].min() - 1, x_data_r[:, 1].max() + 1
        x1x1, x2x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))

        # Predict for all points in the grid
        Z = self.clf.predict(np.c_[x1x1.ravel(), x2x2.ravel()]) 
        Z = Z.reshape(x1x1.shape)

        # Plot
        ax = plt.figure().add_subplot(111)
        
        # Color any pont in the grid based on the value of Z (prediction)
        colors = np.array(['#799FFA', '#ffff00', '#5fff4a', '#f781bf'])
        ax.contourf(x1x1, x2x2,     Z, cmap=matplotlib.colors.ListedColormap(colors))
        ax.axis('off')

        # Plot the training points
        ax.scatter(x_data_r[:,0], x_data_r[:,1], c=self.y_data, cmap=matplotlib.colors.ListedColormap(colors), edgecolor='black', s=20)

        # Title
        f1, f2 = self.x_data.columns[1:]
        plt.title(f'Plotting {f1} (x) vs {f2} (y)', fontsize=10)
        
        plt.savefig(f'../../Saved/{self.filename}.png', dpi=200, bbox_inches='tight')
        if show:    plt.show()
        else:       plt.close()
    
    def double_whammy(self, animated=True, useOld=True):
        '''
        Create a gif that shows both PCA and UMAP dimensionality reduction techniques by:
        1 - Checking if the gifs already exist or if they need to be created
        2 - Reading the two gifs and converting to numpy arrays and adjust their sizes
        3 - Concatenating them along the width axis (horizontally)
        4 - Saving and displaying the gif
        '''
        # If they are not saved and useOld is true then read them
        if not os.path.exists(f'../../Saved/{self.filename}.gif') or not useOld:
            self.illustrate_features_3D(show=False)
        if not os.path.exists(f'../../Saved/{self.filename}.png') or not useOld:
            self.illustrate_features_2D(show=False)
            
        # Read the two gifs
        gif1 = imageio.mimread(f'../../Saved/{self.filename}.gif', memtest=False)
        gif2 = Image.open(f'../../Saved/{self.filename}.png')
        gif1 = np.array([frame[:, :, :3] for frame in gif1])  # remove alpha channel
        gif2 = np.array(gif2)
                
        # Adjusting the size of the two gifs (must have same height) and number of frames
        h1 = gif1.shape[1]                      # height of the 3d gif
        h2, w2 = gif2.shape[:2]                 # height and width of the 2d gif
        wnew = int(w2 * h1 / h2)                # new width of the 2d gif after h2 = h1
        gif2 = Image.fromarray(gif2)
        # Pad it with black pixels so it's not too big
        gif2 = ImageOps.expand(gif2, border=60, fill='black')
        # Now do the resizing
        gif2 = gif2.resize((wnew, h1), resample=Image.Resampling.BICUBIC)   
        gif2 = np.array(gif2)

        # repeat the 2d image along a new axis which will be 0 so it matches the 3d image
        gif2 = np.repeat(gif2[np.newaxis, ...], gif1.shape[0], axis=0)

        # Concatenate the two gifs along the width axis
        gif = np.concatenate((gif1, gif2[..., :3]), axis=2)  # remove alpha channel from gif2

        # Save and display the gif
        imageio.mimsave(f'../../Saved/{self.filename}-D.gif', gif, fps=self.fps if animated else 0.1)
        display(Im(f'../../Saved/{self.filename}-D.gif'))