from .truncnormkde import compute_bandwidth, BoundedKDE
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
import jax.numpy as jnp
import numpy as np

class BoundedKDEPlot:
    def __init__(self, dataframe, prior=None, latex_labels=None):
        self.df = dataframe
        self.priordf = prior
        self.columns = self.df.columns
        self.latex_labels = latex_labels
        if latex_labels is not None:
            self.columns = [latex_labels[col] for col in self.columns]
            
        self.X = self.df.values
        self.X_prior = None
        if self.priordf is not None:
            self.X_prior = self.priordf.values
            
    def compute_values_on_grid(self, X, a=[0,0], b=[1,1], gridsize=100):
        a = jnp.array(a)
        b = jnp.array(b)
        bandwidth = compute_bandwidth(X) # Uses the scotts rule for computation

        # Generate the evaluate grid
        x,y = jnp.linspace(a[0],b[0],gridsize), jnp.linspace(a[1],b[1],gridsize)
        x_2d, y_2d = jnp.meshgrid(x,y)
        X_grid = jnp.stack([x_2d, y_2d],axis=-1)

        # Define the object
        KDE = BoundedKDE(a=a, b=b, bandwidth=bandwidth)

        # Evaluate the KDE on the grid
        computed_values = KDE(X_grid, X)
        return computed_values, X_grid, x_2d, y_2d
        
    def plot_on_axis(self, computed_values, X_grid , x_2d, y_2d, ax, quantiles=[0.9, 0.5], a=[0,0], b=[1,1], gridsize=100, cmax=None, cmin=0, title=None, 
                    quantile_labels=None):
        self.cmax = cmax or computed_values.max()

        levels = np.linspace(cmin, self.cmax, 80)
        cont = ax.contourf(x_2d, y_2d, computed_values, cmap=cm.coolwarm, levels=levels)

        ax.set_xlim(a[0],b[0])
        ax.set_ylim(a[1],b[1])

        ax.set_aspect('equal')
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel(self.columns[0])
        ax.set_ylabel(self.columns[1])
        
        from scipy import interpolate
        ts = np.linspace(0,computed_values.max(),100)
        quantils = ((computed_values[:,:,None] > ts[None, None, :]) * computed_values[:,:,None]).sum(axis=(0,1)) / (gridsize**2)

        f = interpolate.interp1d(quantils, ts)
        t_contours = f(np.array(quantiles))
        
        quantile_labels = [f"{np.round(q*100,0)} %" for q in quantiles]

        fmt = {}
        for name, contour in zip(quantile_labels, t_contours):
            fmt[contour] = name

        quantile_plot = ax.contour(x_2d, y_2d, computed_values, colors="k", alpha=0.3, levels=t_contours)
        ax.clabel(quantile_plot, quantile_plot.levels, inline=True, fmt=fmt, fontsize=10)

        return cont, ax, self.cmax
            
    def plot_posterior(self, quantiles=[0.9, 0.5], a=[0,0], b=[1,1], gridsize=100, cmax=None, title=None, dpi=150):
        fig, axes = plt.subplots(ncols=1, figsize=(5,5), dpi=dpi)
        ax=axes
        computed_values, X_grid, x_2d, y_2d = self.compute_values_on_grid(self.X, a=a, b=b, gridsize=gridsize)
        cont, ax, self.cmax = self.plot_on_axis(computed_values, X_grid, x_2d, y_2d, ax=ax, quantiles=quantiles, a=a, b=b, gridsize=gridsize, cmax=cmax, title=title)
        
        fig.colorbar(cont, shrink=0.8, ticks=np.linspace(0,int(self.cmax),int(self.cmax+1)))
        
        plt.show()
        return fig

    def plot_both(self, quantiles=[0.9, 0.5], a=[0,0], b=[1,1], gridsize=100, cmax=None, title=None, dpi=150):
        fig, axes = plt.subplots(ncols=2, figsize=(10,5), dpi=dpi)
        
        computed_values1, X_grid1, x_2d, y_2d = self.compute_values_on_grid(self.X, a=a, b=b, gridsize=gridsize)
        computed_values2, X_grid2, x_2d, y_2d = self.compute_values_on_grid(self.X_prior, a=a, b=b, gridsize=gridsize)
        
        self.cmax = max(computed_values1.max(), computed_values2.max())
        cmax_int = int(np.ceil(self.cmax))
        
        ax=axes[0]
        
        if title is not None:
            title1 = title[0]
            title2 = title[1]
        else:
            title1 = "posterior"
            title2 = "prior"
        
        cont1, ax, cmax1 = self.plot_on_axis(computed_values1, X_grid1, x_2d, y_2d, ax, quantiles, a, b, gridsize, cmax=cmax_int, title=title1)
        
        ax=axes[1]
        cont2, ax, cmax2 = self.plot_on_axis(computed_values2, X_grid2, x_2d, y_2d, ax, quantiles, a, b, gridsize, cmax=cmax_int, title=title2)
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.84, 0.17, 0.01, 0.62])
        fig.colorbar(cont2, cax=cbar_ax, ticks=np.linspace(0,cmax_int,cmax_int+1))
        
        plt.show()
        return fig
        
    def plot(self, quantiles=[0.9, 0.5], a=[0,0], b=[1,1], gridsize=100, cmax=None, title=None, dpi=150):
        if self.X_prior is not None:
            fig = self.plot_both(quantiles=quantiles, a=a, b=b, gridsize=gridsize, cmax=cmax, title=(title or ['posterior', 'prior']), dpi=dpi)
        else:
            fig = self.plot_posterior(quantiles=quantiles, a=a, b=b, gridsize=gridsize, cmax=cmax, title=title, dpi=dpi)
            
        return fig