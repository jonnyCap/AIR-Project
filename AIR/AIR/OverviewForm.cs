using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace AIR
{
    public partial class OverviewForm : Form
    {
        private SimilarCompaniesForm similarCompaniesForm;
        private StockPredictionForm stockPredictionForm;
        private RankingForm rankingForm;
        private MainForm mainForm;
        public OverviewForm(MainForm mainForm_)
        {
            mainForm = mainForm_;
            similarCompaniesForm = new SimilarCompaniesForm();
            stockPredictionForm = new StockPredictionForm();
            rankingForm = new RankingForm();
            InitializeComponent();
            stockPredictionForm.TopLevel = false;
            similarCompaniesForm.TopLevel = false;
            rankingForm.TopLevel = false;

            panelforms.Controls.Add(stockPredictionForm);
            stockPredictionForm.Dock = DockStyle.Fill;
            panelforms.Controls.Add(similarCompaniesForm);
            similarCompaniesForm.Dock = DockStyle.Fill;
            panelforms.Controls.Add(rankingForm);
            rankingForm.Dock = DockStyle.Fill;
            similarCompaniesForm.Show();



        }
        public void setTickerAndSimilarity(string ticker, string similarity)
        {
            //ticker_list.Add(ticker);
            //similarity_list.Add(similarity);
            similarCompaniesForm.setListinBox(ticker, similarity);

        }
        public void setStockPrediction(JToken stockpred)
        {

            stockPredictionForm.setPlot(stockpred);
        }

        private void closeButton_Click(object sender, EventArgs e)
        {
            Application.Exit();

        }

        private void maximizeButton_Click(object sender, EventArgs e)
        {
            if (this.WindowState == FormWindowState.Normal)
            {
                // Maximize the window if it's in the normal state
                stockPredictionForm.sizeOfPlot(1600, 1200);
                similarCompaniesForm.sizeOfCompanies(1600, 800);
                similarCompaniesForm.changeFontSize(12);

                this.WindowState = FormWindowState.Maximized;
                //RoundPanel(paneltextbox, 30);
            }
            else if (this.WindowState == FormWindowState.Maximized)
            {
                // Restore the window to normal if it's already maximized
                stockPredictionForm.sizeOfPlot(800, 600);
                similarCompaniesForm.sizeOfCompanies(700, 300);
                similarCompaniesForm.changeFontSize(10);

                this.WindowState = FormWindowState.Normal;
                //RoundPanel(paneltextbox, 30);


            }
        }

        private void minimizeButton_Click(object sender, EventArgs e)
        {
            this.WindowState = FormWindowState.Minimized;

        }

        private void buttonStock_Click(object sender, EventArgs e)
        {
            stockPredictionForm.Show();
            similarCompaniesForm.Hide();
            rankingForm.Hide();
        }

        private void buttonCompanies_Click(object sender, EventArgs e)
        {
            similarCompaniesForm.Show();
            rankingForm.Hide();
            stockPredictionForm.Hide();

        }

        private void buttonRanking_Click(object sender, EventArgs e)
        {
            rankingForm.Show();
            similarCompaniesForm.Hide();
            stockPredictionForm.Hide();



        }
        private bool isDragging = false; // To track if the form is being dragged
        private Point dragCursorPoint; // To hold the current mouse position
        private Point dragFormPoint; // To hold the initial position of the form


        private void closepanel_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                isDragging = true; // Set dragging flag
                dragCursorPoint = Cursor.Position; // Get the current mouse position
                dragFormPoint = this.Location; // Get the current form location
            }
        }

        private void closepanel_MouseUp(object sender, MouseEventArgs e)
        {
            // Release dragging flag when mouse button is released
            if (e.Button == MouseButtons.Left)
            {
                isDragging = false; // Stop dragging
            }
        }

        private void closepanel_MouseMove(object sender, MouseEventArgs e)
        {
            if (isDragging)
            {
                Point dif = Point.Subtract(Cursor.Position, new Size(dragCursorPoint));
                this.Location = Point.Add(dragFormPoint, new Size(dif));
            }
        }

        private void buttonnewidea_Click(object sender, EventArgs e)
        {
            this.Hide();
            similarCompaniesForm.clearListBox();
            stockPredictionForm.clearPlot();
            mainForm.ClearTextBox();
            mainForm.Show();
            mainForm.WindowState = this.WindowState;
            similarCompaniesForm.Show();
            stockPredictionForm.Hide();
            rankingForm.Hide();
            
            
        }
    }
}
