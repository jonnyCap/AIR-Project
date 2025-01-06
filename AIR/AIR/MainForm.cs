using Microsoft.VisualBasic.ApplicationServices;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace AIR
{
    public partial class MainForm : Form
    {
        private OverviewForm overviewForm;
        public MainForm()
        {
            overviewForm = new OverviewForm(this);
            InitializeComponent();
            progressBar.Visible = false;
            RoundPanel(paneltextbox, 30);


        }
        public void ClearTextBox()
        {
            textBoxidea.Clear();
            prediction_list.Clear();
            company_stock.Clear();
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
        private void closeButton_Click(object sender, EventArgs e)
        {
            Application.Exit();

        }

        private void minimizeButton_Click(object sender, EventArgs e)
        {
            RoundPanel(paneltextbox, 30);
            this.WindowState = FormWindowState.Minimized;
        }

        private void maximizeButton_Click(object sender, EventArgs e)
        {
            if (this.WindowState == FormWindowState.Normal)
            {
                // Maximize the window if it's in the normal state
                this.WindowState = FormWindowState.Maximized;
                RoundPanel(paneltextbox, 30);
                
            }
            else if (this.WindowState == FormWindowState.Maximized)
            {
                // Restore the window to normal if it's already maximized
                this.WindowState = FormWindowState.Normal;
                RoundPanel(paneltextbox, 30);


            }
        }

        private void labeltitle_Click(object sender, EventArgs e)
        {

        }

        private void label1_Click(object sender, EventArgs e)
        {

        }
        JArray similarCompanies;
        private void getSimilarCompanies(string idea, Dictionary<string, List<double>> stock_per_company)
        {

            string projectRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", ".."));
            //Debug.WriteLine($"ProjectRootCompanies: {projectRoot}");

            string pythonScript = Path.Combine(projectRoot, "RetrievalSystem", "RetrievalSystem.py");
            string pythonInterpreter = Path.Combine(projectRoot, "venv", "Scripts", "python.exe");
            try
            {
                // Start the Python process
                var start = new ProcessStartInfo
                {
                    FileName = pythonInterpreter,
                    Arguments = $"\"{pythonScript}\" --idea \"{idea}\" --top_n 10", // Optional: Add top_n if needed
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using (var process = Process.Start(start))
                {
                    using (var reader = process.StandardOutput)
                    {
                        string result = reader.ReadToEnd();
                        process.WaitForExit();
                        //Debug.WriteLine(result);

                        // Parse and display the result
                        similarCompanies = JArray.Parse(result.Trim());
                        //Debug.WriteLine(similarCompanies.ToString());
                        //stock_per_company = new Dictionary<string, List<double>>();
                        stock_per_company.Clear();
                        foreach (var company in similarCompanies)
                        {
                            if (company != null)
                            {
                                List<double> stock = new List<double>();
                                for (int i = 1; i <= 24; i++)
                                {
                                    stock.Add(double.Parse(company["month_" + i + "_performance"].ToString()));
                                }

                                // Add data to the provided stock_per_company dictionary
                                stock_per_company[company["business_description"].ToString()] = stock;
                            }
                        }
                        
                        //Debug.WriteLine($"hallo: {string.Join(", ", stock_per_company.Keys)}");


                    }

                    // Check for errors
                    using (var errorReader = process.StandardError)
                    {
                        string errors = errorReader.ReadToEnd();
                        if (!string.IsNullOrEmpty(errors))
                        {
                            MessageBox.Show(errors, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex);
            }
        }
        JArray stockPrediction;
        private void getStockPrediction(string idea, List<double> pred_list)
        {
            string projectRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", ".."));
            //Debug.WriteLine($"ProjectRoot: {projectRoot}");
            string pythonScript = Path.Combine(projectRoot, "PredictionModel", "RetrievalAugmentedPredictionModel.py");
            string pythonInterpreter = Path.Combine(projectRoot, "venv", "Scripts", "python.exe");
            try
            {
                // Start the Python process
                var start = new ProcessStartInfo
                {
                    FileName = pythonInterpreter,
                    Arguments = $"\"{pythonScript}\" --idea \"{idea}\"", // Optional: Add top_n if needed
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using (var process = Process.Start(start))
                {
                    using (var reader = process.StandardOutput)
                    {
                        string result = reader.ReadToEnd();

                        process.WaitForExit();

                        //process.ErrorDataReceived += (sender, e) => Debug.WriteLine("Error: " + e.Data);
                        //process.BeginErrorReadLine();
                        //process.WaitForExit();
                        // Parse and display the result
                        //Debug.WriteLine($"Result-Stock: {result}");

                        stockPrediction = JArray.Parse(result);

                        foreach (var predictions in stockPrediction)
                        {
                            foreach (var prediction in predictions)
                            { 
                                double val = double.Parse(prediction.ToString());
                                pred_list.Add(val);
                            }
                        }
                        // Display the results - Example code
                    }
                    // Check for errors
                    using (var errorReader = process.StandardError)
                    {
                        string errors = errorReader.ReadToEnd();
                        if (!string.IsNullOrEmpty(errors))
                        {
                            MessageBox.Show(errors, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex);
            }
        }
        JArray rankings;
        private void getRanking(string idea, List<double> predlist, Dictionary<string, List<double>> company_stock)
        {
            string projectRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", ".."));
            //Debug.WriteLine($"ProjectRoot: {projectRoot}");
            string pythonScript = Path.Combine(projectRoot, "RankingSystem", "RankingModel.py");
            string pythonInterpreter = Path.Combine(projectRoot, "venv", "Scripts", "python.exe");
            company_stock[idea] = predlist;
            try
            {
                string companyStockJson = JsonConvert.SerializeObject(company_stock);


                string escapedCompanyStockJson = Uri.EscapeDataString(companyStockJson);
                //Debug.WriteLine($"Serialized company stock JSON: {companyStockJson}");

                // Start the Python process
                var start = new ProcessStartInfo
                {
                    FileName = pythonInterpreter,
                    Arguments = $"\"{pythonScript}\" --company_and_stock \"{escapedCompanyStockJson}\"", // Optional: Add top_n if needed
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using (var process = Process.Start(start))
                {
                    using (var reader = process.StandardOutput)
                    {
                        string result = reader.ReadToEnd();
                        //process.ErrorDataReceived += (sender, e) => Debug.WriteLine("Error: " + e.Data);
                        //process.BeginErrorReadLine();
                        //process.WaitForExit();
                        process.WaitForExit();
                        // Parse and display the result
                        Debug.WriteLine($"{result}");

                        rankings = JArray.Parse(result);
                        // Display the results - Example code
                    }
                    // Check for errors
                    using (var errorReader = process.StandardError)
                    {
                        string errors = errorReader.ReadToEnd();
                        if (!string.IsNullOrEmpty(errors))
                        {
                            MessageBox.Show(errors, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex);
            }
        }
        //async because otherwise it blocks other functions
        Dictionary<string, List<double>> company_stock = new Dictionary<string, List<double>>();
        List<double> prediction_list = new List<double>();

        private async void iconButtonAnalyze_Click(object sender, EventArgs e)
        {
            try
            {
                progressBar.Visible = true; // Show progress bar
                progressBar.Style = ProgressBarStyle.Marquee; // Optional: Indeterminate style

                string idea = textBoxidea.Text; // Get the text from the textbox
                if (string.IsNullOrWhiteSpace(idea))
                {
                    MessageBox.Show("Please enter an idea.", "Input Required", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    progressBar.Visible = false; // Hide progress bar
                    return;
                }

                // Run time-consuming tasks asynchronously
                var similarCompaniesTask = Task.Run(() => getSimilarCompanies(idea,company_stock));
                var stockPredictionTask = Task.Run(() => getStockPrediction(idea, prediction_list));
                // Wait for both tasks to complete
                
                await Task.WhenAll(similarCompaniesTask, stockPredictionTask);
                
                var rankingTask = Task.Run(() => getRanking(idea, prediction_list, company_stock));
                await Task.WhenAll(rankingTask);
                // Display the results after the tasks complete
                DisplayResultsForSimCompanies(similarCompanies, stockPrediction);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"An error occurred: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                progressBar.Visible = false; // Hide the progress bar after completion
            }
        }

        // This method will handle the display of results
        private void DisplayResultsForSimCompanies(JArray similarCompanies, JArray stockPrediction)
        {

            // Iterate over each entry in the JSON array

            foreach (var company in similarCompanies)
            {
                overviewForm.setTickerAndSimilarity(company["tickers"].ToString(), company["similarity"].ToString());

            }
            foreach (var pred in stockPrediction)
            {
                overviewForm.setStockPrediction(pred);
            }
            foreach (var rank in rankings)
            {
                string ticker = null;
                foreach(var company in similarCompanies)
                {
                    if (company["business_description"].ToString() == rank["idea"].ToString())
                    {
                        ticker = company["tickers"].ToString();
                        break;
                    }
                }
                if (ticker == null)
                {
                    ticker = "Your Idea";
                }
                overviewForm.setRankingCompanies(rank["idea"].ToString(),ticker, double.Parse(rank["score"].ToString()));
            }
            overviewForm.Show();
            this.Hide();
        }
        private bool isDragging = false; // To track if the form is being dragged
        private Point dragCursorPoint; // To hold the current mouse position
        private Point dragFormPoint; // To hold the initial position of the form
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
                Point dif = Point.Subtract(Cursor.Position, new Size(dragCursorPoint)); // Calculate the difference
                this.Location = Point.Add(dragFormPoint, new Size(dif)); // Move the form
            }
        }

        private void closepanel_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                isDragging = true; // Set dragging flag
                dragCursorPoint = Cursor.Position; // Get the current mouse position
                dragFormPoint = this.Location; // Get the current form location
            }
        }
    }
}

