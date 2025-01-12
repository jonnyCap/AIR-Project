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
using System.Globalization;
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
        class CompanyStockData
        {
            public string CompanyName { get; set; }
            public List<double> StockPerformance { get; set; }
            public List<double> Embedding { get; set; }
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
        private void getSimilarCompanies(string idea, List<CompanyStockData> companyStockDataList, List<string> ideas)
        {
            string projectRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", ".."));
            string pythonModule = "RetrievalSystem.RetrievalSystem_frontend";
            string pythonInterpreter = Path.Combine(projectRoot, "venv", "Scripts", "python.exe");

            try
            {
                // Start the Python process
                var start = new ProcessStartInfo
                {
                    FileName = pythonInterpreter,
                    Arguments = $"-m {pythonModule} --idea \"{idea}\" --top_n 10", // Optional: Add top_n if needed
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = projectRoot
                };

                using (var process = Process.Start(start))
                {
                    using (var reader = process.StandardOutput)
                    {
                        string result = reader.ReadToEnd();
                        process.WaitForExit();
                       
                        result = result.Replace("Loading RAP model weights...", "");

                        // Parse and display the result
                        similarCompanies = JArray.Parse(result.Trim());
                        //Debug.WriteLine($"Was los: {similarCompanies}");

                        foreach (var company in similarCompanies)
                        {
                            //Debug.WriteLine($"^Company: {company}");

                            List<string> numberStrings = company["embedding"].ToString().Trim('[', ']').Split(',').ToList();
                            List<double> embeddings = new List<double>();

                            foreach (string item in numberStrings)
                            {
                                if (!string.IsNullOrWhiteSpace(item))
                                {
                                    try
                                    {
                                        double parsedValue = double.Parse(item, CultureInfo.InvariantCulture);
                                        embeddings.Add(parsedValue);
                                    }
                                    catch (FormatException ex)
                                    {
                                        Debug.WriteLine($"Failed to parse '{item}': {ex.Message}");
                                    }
                                }
                                else
                                {
                                    Debug.WriteLine($"Skipping invalid item: '{item}'");
                                }
                            }

                            if (company != null)
                            {
                                List<double> stock = new List<double>();

                                if (company["tickers"].ToString() != "new_idea")
                                {
                                    for (int i = 1; i <= 12; i++)
                                    {
                                        stock.Add(double.Parse(company["month_" + i + "_performance"].ToString()));
                                    }
                                }

                                // Create a new CompanyStockData object and add it to the list
                                companyStockDataList.Add(new CompanyStockData
                                {
                                    CompanyName = company["business_description"].ToString(),
                                    StockPerformance = stock,
                                    Embedding = embeddings
                                });

                                // Add the company description to the ideas list
                                ideas.Add(company["business_description"].ToString());
                            }
                        }
                        //var companyData = companyStockDataList.FirstOrDefault(c => c.CompanyName.Equals(idea, StringComparison.OrdinalIgnoreCase));
                        //Debug.WriteLine($"{string.Join(", ", companyData.Embedding)}");


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
        private void getStockPrediction(List<string> ideas, List<double> pred_list)
        {
            string projectRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", ".."));
            //Debug.WriteLine($"ProjectRoot: {projectRoot}");
            string pythonModule = "PredictionModel.RetrievalAugmentedPredictionModel_frontend";
            string pythonInterpreter = Path.Combine(projectRoot, "venv", "Scripts", "python.exe");
            //Debug.WriteLine($"ideas: {ideas[0]} {ideas[1]}");
            //foreach(var idea in ideas)
            //{
            //    Debug.WriteLine($"idea: {idea}");
            //}
            string ideasJson = JsonConvert.SerializeObject(ideas);
            string escapedideasJson = Uri.EscapeDataString(ideasJson);


            try
            {
                // Start the Python process
                var start = new ProcessStartInfo
                {
                    FileName = pythonInterpreter,
                    Arguments = $" -m {pythonModule} --ideas \"{escapedideasJson}\"", // Optional: Add top_n if needed
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = projectRoot

                };

                using (var process = Process.Start(start))
                {
                    using (var reader = process.StandardOutput)
                    {
                        string result = reader.ReadToEnd();

                        //process.WaitForExit();

                        process.ErrorDataReceived += (sender, e) => Debug.WriteLine("Error: " + e.Data);
                        process.BeginErrorReadLine();
                        process.WaitForExit();
                        // Parse and display the result
                        result = result.Replace("Loading RAP model weights...", "");

                        //Debug.WriteLine($"Result-Stock: {result}");

                        stockPrediction = JArray.Parse(result);

                        foreach (var prediction in stockPrediction)
                        {
                             
                            double val = double.Parse(prediction.ToString());
                            pred_list.Add(val);
                            
                        }
                        // Display the results - Example code
                    }
                    //Debug.WriteLine($"pred: {pred_list[0]} {pred_list[1]}");

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
        private void getRanking(string idea, List<double> predlist,List<CompanyStockData> company_stock)
        {
            string projectRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", ".."));
            //Debug.WriteLine($"Test: {company_stock}");
            string pythonModule = "RankingModel.RankingModel_frontend";
            string pythonInterpreter = Path.Combine(projectRoot, "venv", "Scripts", "python.exe");
            var yourcompany = company_stock.FirstOrDefault(c => c.CompanyName == idea);
            yourcompany.StockPerformance = predlist;
            try
            {


                string json = JsonConvert.SerializeObject(company_stock, Formatting.Indented);

                // Write to file
                string path = Path.Combine(Directory.GetCurrentDirectory(), "company_stock_data.json");
                File.WriteAllText(path, json);
                // Start the Python process
                var start = new ProcessStartInfo
                {
                    FileName = pythonInterpreter,
                    Arguments = $"-m {pythonModule} --company_and_stock \"{path}\"", // Optional: Add top_n if needed
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = projectRoot

                };

                using (var process = Process.Start(start))
                {
                    using (var reader = process.StandardOutput)
                    {
                        string result = reader.ReadToEnd();
                        process.ErrorDataReceived += (sender, e) => Debug.WriteLine("Error: " + e.Data);
                        process.BeginErrorReadLine();
                        //process.WaitForExit();
                        process.WaitForExit();
                        // Parse and display the result
                        //Debug.WriteLine($"Result-Ranking{result}");
                        result = result.Replace("Loading RAP model weights...", "");

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
        List<CompanyStockData> company_stock = new List<CompanyStockData>();
        List<double> prediction_list = new List<double>();

        List<string> ideas = new List<string>();
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
                var similarCompaniesTask = Task.Run(() => getSimilarCompanies(idea,company_stock,ideas));
                await Task.WhenAll(similarCompaniesTask);

                var stockPredictionTask = Task.Run(() => getStockPrediction(ideas, prediction_list));
                // Wait for both tasks to complete
                
                await Task.WhenAll(stockPredictionTask);

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
                foreach (var company in similarCompanies)
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
                overviewForm.setRankingCompanies(rank["idea"].ToString(), ticker, double.Parse(rank["rating"].ToString()));
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

