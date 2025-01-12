using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace AIR
{
    public partial class SimilarCompaniesForm : Form
    {
        public SimilarCompaniesForm()
        {
            InitializeComponent();
        }

        public void RoundSimCompaniesPanels()
        {
            //InitializeComponent();
            for (int i = 1; i <= 10; i++)
            {
                string panelName = $"panelcompany{i}";
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
        public void setListinBox(string ticker, string similiarity)
        {
            var d_similiarity = Double.Parse(similiarity);
            switch(counter)
            {
                case 1:
                    labelticker1.Text = ticker;
                    labelsimilarity1.Text = "Similiarity: "+ Math.Round(d_similiarity, 4);
                    break;
                case 2:

                    labelticker2.Text = ticker;
                    labelsimilarity2.Text = "Similiarity: " + Math.Round(d_similiarity,4);
                    break;
                case 3:

                    labelticker3.Text = ticker;
                    labelsimilarity3.Text = "Similiarity: " + Math.Round(d_similiarity, 4);
                    break;
                case 4:
                    labelticker4.Text = ticker;
                    labelsimilarity4.Text = "Similiarity: " + Math.Round(d_similiarity, 4);
                    break;

                case 5:
                    labelticker5.Text = ticker;
                    labelsimilarity5.Text = "Similiarity: " + Math.Round(d_similiarity, 4);
                    break;

                case 6:
                    labelticker6.Text = ticker;
                    labelsimilarity6.Text = "Similiarity: " + Math.Round(d_similiarity, 4);
                    break;
                case 7:
                    labelticker7.Text = ticker;
                    labelsimilarity7.Text = "Similiarity: " + Math.Round(d_similiarity, 4);
                    break;
                case 8:
                    labelticker8.Text = ticker;
                    labelsimilarity8.Text = "Similiarity: " + Math.Round(d_similiarity, 4);
                    break;
                case 9:
                    labelticker9.Text = ticker;
                    labelsimilarity9.Text = "Similiarity: " + Math.Round(d_similiarity, 4);
                    break;
                case 10:
                    labelticker10.Text = ticker;
                    labelsimilarity10.Text = "Similiarity: " + Math.Round(d_similiarity, 4);
                    break;

            }
            counter++;

        }
        //public void sizeOfCompanies(int x, int y)
        //{
        //    listofcompanies.Size = new Size(x, y);
        //}
        //public void changeFontSize(int size)
        //{
        //    listofcompanies.Font = new Font("Segoe UI", size, FontStyle.Regular);
        //}
        //public void clearListBox()
        //{
        //    labelsimilarity1.Text.
        //}
        private void listofcompanies_SelectedIndexChanged(object sender, EventArgs e)
        {

        }
    }
}
