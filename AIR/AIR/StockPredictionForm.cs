using Newtonsoft.Json.Linq;
using OxyPlot.Series;
using OxyPlot.WindowsForms;
using OxyPlot;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Diagnostics;
using OxyPlot.Axes;

namespace AIR
{
    public partial class StockPredictionForm : Form
    {
        public StockPredictionForm()
        {
            InitializeComponent();
        }

        private void textBoxidea_TextChanged(object sender, EventArgs e)
        {

        }
        List<double> pred_list = new List<double>();
        public void setPlot(JToken prediction)
        {
            
            double val = double.Parse(prediction.ToString());
            pred_list.Add(val);

            var plotModel = new PlotModel
            {
                Title = "Visualization of Final Predictions",
                TitleColor = OxyColors.White, // Set the title text color
                Background = OxyColor.FromRgb(65, 65, 65) // Set the background color
            };

            // Create a LineSeries for the predictions
            var lineSeries = new LineSeries
            {
                Title = "Final Predictions",
                MarkerType = MarkerType.Circle,
                MarkerSize = 4,
                Color = OxyColors.Red // Set the line color to red
            };

            // Add data points to the line series
            for (int i = 0; i < pred_list.Count; i++)
            {
                lineSeries.Points.Add(new DataPoint(i, pred_list[i]));
            }

            // Add the line series to the plot model
            plotModel.Series.Add(lineSeries);

            // Customize the axes
            plotModel.Axes.Add(new LinearAxis
            {
                Position = AxisPosition.Bottom,
                Title = "Months", // X-axis title
                TitleColor = OxyColors.White, // X-axis title color
                TextColor = OxyColors.White, // X-axis tick text color
                AxislineColor = OxyColor.FromRgb(85, 85, 85), // X-axis line color
                MajorGridlineColor = OxyColor.FromRgb(85, 85, 85), // X-axis major gridline color
                MinorGridlineColor = OxyColor.FromRgb(85, 85, 85), // X-axis minor gridline color
                MajorGridlineStyle = LineStyle.Solid,
                MinorGridlineStyle = LineStyle.Dot
            });

            plotModel.Axes.Add(new LinearAxis
            {
                Position = AxisPosition.Left,
                Title = "Performance", // Y-axis title
                TitleColor = OxyColors.White, // Y-a    xis title color
                TextColor = OxyColors.White, // Y-axis tick text color
                AxislineColor = OxyColor.FromRgb(85, 85, 85), // Y-axis line color
                MajorGridlineColor = OxyColor.FromRgb(85, 85, 85), // Y-axis major gridline color
                MinorGridlineColor = OxyColor.FromRgb(85, 85, 85), // Y-axis minor gridline color
                MajorGridlineStyle = LineStyle.Solid,
                MinorGridlineStyle = LineStyle.Dot
            });

            // Set the PlotView's model and background color
            plotViewStock.Model = plotModel;
            plotViewStock.BackColor = Color.FromArgb(65, 65, 65);


        }
        public void sizeOfPlot(int x, int y)
        {
            plotViewStock.Size = new Size(x, y);
        }
        public void clearPlot() 
        {
            pred_list.Clear();
        }
        private void plotViewStock_Click(object sender, EventArgs e)
        {

        }
    }
}
