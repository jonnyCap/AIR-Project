namespace AIR
{
    partial class SimilarCompaniesForm
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
            label = new Label();
            listofcompanies = new ListBox();
            tableLayoutPanel1.SuspendLayout();
            SuspendLayout();
            // 
            // tableLayoutPanel1
            // 
            tableLayoutPanel1.ColumnCount = 2;
            tableLayoutPanel1.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));
            tableLayoutPanel1.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));
            tableLayoutPanel1.Controls.Add(label, 0, 0);
            tableLayoutPanel1.Controls.Add(listofcompanies, 0, 1);
            tableLayoutPanel1.Dock = DockStyle.Fill;
            tableLayoutPanel1.Location = new Point(0, 0);
            tableLayoutPanel1.Name = "tableLayoutPanel1";
            tableLayoutPanel1.RowCount = 2;
            tableLayoutPanel1.RowStyles.Add(new RowStyle(SizeType.Percent, 25F));
            tableLayoutPanel1.RowStyles.Add(new RowStyle(SizeType.Percent, 75F));
            tableLayoutPanel1.Size = new Size(1250, 669);
            tableLayoutPanel1.TabIndex = 0;
            // 
            // label
            // 
            label.Anchor = AnchorStyles.Bottom;
            label.AutoSize = true;
            tableLayoutPanel1.SetColumnSpan(label, 2);
            label.Font = new Font("Segoe UI", 11F);
            label.ForeColor = SystemColors.Window;
            label.Location = new Point(490, 96);
            label.Margin = new Padding(3, 0, 3, 30);
            label.Name = "label";
            label.Size = new Size(270, 41);
            label.TabIndex = 0;
            label.Text = "Similiar Companies";
            // 
            // listofcompanies
            // 
            listofcompanies.Anchor = AnchorStyles.Top;
            listofcompanies.BackColor = Color.FromArgb(64, 64, 64);
            listofcompanies.BorderStyle = BorderStyle.None;
            tableLayoutPanel1.SetColumnSpan(listofcompanies, 2);
            listofcompanies.Font = new Font("Segoe UI", 10F);
            listofcompanies.ForeColor = SystemColors.Window;
            listofcompanies.FormattingEnabled = true;
            listofcompanies.ItemHeight = 37;
            listofcompanies.Location = new Point(275, 170);
            listofcompanies.Name = "listofcompanies";
            listofcompanies.Size = new Size(700, 296);
            listofcompanies.TabIndex = 1;
            listofcompanies.SelectedIndexChanged += listofcompanies_SelectedIndexChanged;
            // 
            // SimilarCompaniesForm
            // 
            AutoScaleDimensions = new SizeF(13F, 32F);
            AutoScaleMode = AutoScaleMode.Font;
            BackColor = Color.FromArgb(50, 50, 50);
            ClientSize = new Size(1250, 669);
            Controls.Add(tableLayoutPanel1);
            FormBorderStyle = FormBorderStyle.None;
            Name = "SimilarCompaniesForm";
            Text = "SimilarCompaniesForm";
            tableLayoutPanel1.ResumeLayout(false);
            tableLayoutPanel1.PerformLayout();
            ResumeLayout(false);
        }

        #endregion

        private TableLayoutPanel tableLayoutPanel1;
        private Label label;
        private ListBox listofcompanies;
    }
}