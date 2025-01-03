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
        public void setPlot(JToken predictions)
        {
            foreach (var prediction in predictions)
            {
                double val = double.Parse(prediction.ToString());
                pred_list.Add(val);
            }
            var plotModel = new PlotModel { Title = "Visualization of Final Predictions" };
            var lineSeries = new LineSeries
            {
                Title = "Final Predictions",
                MarkerType = MarkerType.Circle,
                MarkerSize = 4
            };



            for (int i = 0; i < pred_list.Count; i++)
            {
                lineSeries.Points.Add(new DataPoint(i, pred_list[i]));
            }

            plotModel.Series.Add(lineSeries);
            //plotModel.
            plotViewStock.Model = plotModel;
            plotViewStock.BackColor = Color.White;


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
