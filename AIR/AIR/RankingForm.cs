using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics.Metrics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace AIR
{
    public partial class RankingForm : Form
    {
        public RankingForm()
        {
            InitializeComponent();
        }
        public void RoundRankingPanels()
        {
            //InitializeComponent();
            for (int i = 1; i <= 11; i++)
            {
                string panelName = $"panelranking{i}";
                System.Windows.Forms.Control panel = this.Controls.Find(panelName, true).FirstOrDefault();
                if (panel != null && panel is Panel)
                {
                    RoundPanel((Panel)panel, 30);
                }
            }
        }
        private void RoundPanel(Panel panel, int cornerRadius)
        {
            // Ensure the panel has a valid size before creating the region
            if (panel.Width == 0 || panel.Height == 0)
                return;

            // Create a graphical path
            GraphicsPath path = new GraphicsPath();

            // Define rounded corners
            path.StartFigure();
            path.AddArc(new Rectangle(0, 0, cornerRadius, cornerRadius), 180, 90);  // Top left
            path.AddArc(new Rectangle(panel.Width - cornerRadius, 0, cornerRadius, cornerRadius), -90, 90); // Top right
            path.AddArc(new Rectangle(panel.Width - cornerRadius, panel.Height - cornerRadius, cornerRadius, cornerRadius), 0, 90); // Bottom right
            path.AddArc(new Rectangle(0, panel.Height - cornerRadius, cornerRadius, cornerRadius), 90, 90); // Bottom left
            path.CloseFigure();

            // Set the region of the panel to the rounded shape
            panel.Region = new Region(path);
        }
        int counter = 1;
        public void setRankedCompanies(string idea, string ticker, double score)
        {
            switch (counter)
            {
                case 1:
                    labelticker1.Text = $"{counter}: " + ticker;
                    labelscore1.Text = "score: " + Math.Round(score, 4);
                    break;
                case 2:

                    labelticker2.Text = $"{counter}: " + ticker;
                    labelscore2.Text = "score: " + Math.Round(score, 4);
                    break;
                case 3:

                    labelticker3.Text = $"{counter}: " + ticker;
                    labelscore3.Text = "score: " + Math.Round(score, 4);
                    break;
                case 4:
                    labelticker4.Text = $"{counter}: " + ticker;
                    labelscore4.Text = "score: " + Math.Round(score, 4);
                    break;

                case 5:
                    labelticker5.Text = $"{counter}: " + ticker;
                    labelscore5.Text = "score: " + Math.Round(score, 4);
                    break;

                case 6:
                    labelticker6.Text = $"{counter}: " + ticker;
                    labelscore6.Text = "score: " + Math.Round(score, 4);
                    break;
                case 7:
                    labelticker7.Text = $"{counter}: " + ticker;
                    labelscore7.Text = "score: " + Math.Round(score, 4);
                    break;
                case 8:
                    labelticker8.Text = $"{counter}: " + ticker;
                    labelscore8.Text = "score: " + Math.Round(score, 4);
                    break;
                case 9:
                    labelticker9.Text = $"{counter}: " + ticker;
                    labelscore9.Text = "score: " + Math.Round(score, 4);
                    break;
                case 10:
                    labelticker10.Text = $"{counter}: " + ticker;
                    labelscore10.Text = "score: " + Math.Round(score, 4);
                    break;
                case 11:
                    labelticker11.Text = $"{counter}: " + ticker;
                    labelscore11.Text = "score: " + Math.Round(score, 4);
                    break;

            }
            counter++;
        }

        private void tableLayoutPanel1_Paint(object sender, PaintEventArgs e)
        {

        }

        private void tableLayoutPanel7_Paint(object sender, PaintEventArgs e)
        {

        }
    }


}
