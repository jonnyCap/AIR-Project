namespace AIR
{
    partial class StockPredictionForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            tableLayoutPanel1 = new TableLayoutPanel();
            plotViewStock = new OxyPlot.WindowsForms.PlotView();
            tableLayoutPanel1.SuspendLayout();
            SuspendLayout();
            // 
            // tableLayoutPanel1
            // 
            tableLayoutPanel1.ColumnCount = 2;
            tableLayoutPanel1.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));
            tableLayoutPanel1.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));
            tableLayoutPanel1.Controls.Add(plotViewStock, 0, 0);
            tableLayoutPanel1.Dock = DockStyle.Fill;
            tableLayoutPanel1.Location = new Point(0, 0);
            tableLayoutPanel1.Name = "tableLayoutPanel1";
            tableLayoutPanel1.RowCount = 1;
            tableLayoutPanel1.RowStyles.Add(new RowStyle(SizeType.Percent, 50F));
            tableLayoutPanel1.RowStyles.Add(new RowStyle(SizeType.Percent, 50F));
            tableLayoutPanel1.Size = new Size(1250, 669);
            tableLayoutPanel1.TabIndex = 0;
            // 
            // plotViewStock
            // 
            plotViewStock.Anchor = AnchorStyles.None;
            tableLayoutPanel1.SetColumnSpan(plotViewStock, 2);
            plotViewStock.Location = new Point(225, 34);
            plotViewStock.Name = "plotViewStock";
            plotViewStock.PanCursor = Cursors.Hand;
            plotViewStock.Size = new Size(800, 600);
            plotViewStock.TabIndex = 0;
            plotViewStock.Text = "plotView1";
            plotViewStock.ZoomHorizontalCursor = Cursors.SizeWE;
            plotViewStock.ZoomRectangleCursor = Cursors.SizeNWSE;
            plotViewStock.ZoomVerticalCursor = Cursors.SizeNS;
            plotViewStock.Click += plotViewStock_Click;
            // 
            // StockPredictionForm
            // 
            AutoScaleDimensions = new SizeF(13F, 32F);
            AutoScaleMode = AutoScaleMode.Font;
            BackColor = Color.FromArgb(50, 50, 50);
            ClientSize = new Size(1250, 669);
            Controls.Add(tableLayoutPanel1);
            FormBorderStyle = FormBorderStyle.None;
            Name = "StockPredictionForm";
            Text = "StockPredictionForm";
            tableLayoutPanel1.ResumeLayout(false);
            ResumeLayout(false);
        }

        #endregion

        private TableLayoutPanel tableLayoutPanel1;
        private OxyPlot.WindowsForms.PlotView plotViewStock;
    }
}