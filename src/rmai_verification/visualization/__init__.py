import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging
import numpy as np

LOG = logging.getLogger(__name__)

def plot_overview(dataset, prefix, x="lead_time", y=None, hue="model", avg_dims=None, avg_method="mean", **kwargs):
    total_plots = int(len(dataset["metric"]))
    cols_per_page = 2 #min(total_plots, 2)
    rows_per_page = 2 #(int(total_plots/cols_per_page), 1)
    plots_per_page = rows_per_page * cols_per_page
    total_pages = -(-total_plots // plots_per_page)


    if y in ["latitude", ]:
        sharey = True
    else:
        sharey = False

    if x == "lead_time":
        xticks = dataset[x].values.astype(np.float64)
        xticklabels = (dataset[x].values/np.timedelta64(1,"h")).astype(np.int16)
    
    xlabel = dataset[x].name.replace("_", " ")

    
    for variable in dataset:
        LOG.info(f"Started plotting metrics for variable {variable}")
        data = dataset[variable]
        title = data.name
        path = f"{prefix}_{variable}.pdf"
        with PdfPages(path) as pdf:
            for page in range(total_pages):
                start = page * plots_per_page
                end = min(start + plots_per_page, total_plots)
                fig, axs = plt.subplots(
                    nrows = rows_per_page,
                    ncols = cols_per_page,
                    sharey = sharey,
                    figsize=(17,10),
                    #aspect=16/9,
                    #marker="o",
                    **kwargs
                )
                for i_ax, i_metric in enumerate(range(start,end)):
                    ylabel = data["metric"].values[i_metric]
                    LOG.info(f"Plotting metric: {ylabel}")
                    ax = axs.flat[i_ax]
                    da = data.isel(
                        metric=i_metric
                    )
                    
                    if avg_dims is not None:
                        if avg_method == "mean":
                            da = da.mean(dim=avg_dims)
                        elif avg_method == "median":
                            da = da.median(dim=avg_dims)
                        else:
                            raise ValueError(f"Unknown avg_method: {avg_method}")
                        
                    da.plot(
                        x=x,
                        y=y,
                        hue=hue,
                        marker="o",
                        ax=ax
                    )
                    if x == "lead_time":
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(xticklabels)
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    ax.set_title(title)
                    ax.grid(True, alpha=0.5)
                pdf.savefig(fig)
                plt.close(fig)
