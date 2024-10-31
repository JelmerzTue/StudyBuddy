import matplotlib.pyplot as plt
import matplotlib.animation as animation


class LivePlotter:
    def __init__(self, data_source, data_label='Data', ylim=(-5, 5), history_length=250):
        self.data_source = data_source  # This could be a list or any iterable
        self.data_label = data_label
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_ylim(*ylim)  # Y-axis limits
        self.ax.set_xlim(0, history_length)  # X-axis limits
        self.ax.set_title(f'{data_label} Over Time')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel(data_label)
        self.history_length = history_length

    def init_plot(self):
        # Initialize the background of the plot
        self.line.set_data([], [])
        return self.line,

    def update_plot(self, frame):
        # Update the plot with the current data source values
        # if len(self.data_source) >= self.history_length:
        #     self.data_source = self.data_source[-self.history_length:]  # Keep only the last history_length values
        x = range(len(self.data_source))
        self.line.set_data(x, self.data_source)
        self.ax.grid()
        return self.line,

    def start_plot(self):
        # Start the animation
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot,
                                           interval=50, blit=True, cache_frame_data=False)
        plt.show(block=False)

    def stop_plot(self):
        plt.close(self.fig)
        plt.close('all')
