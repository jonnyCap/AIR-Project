namespace AIR
{
    partial class OverviewForm
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
            closepanel = new Panel();
            labeltitle = new Label();
            minimizeButton = new FontAwesome.Sharp.IconButton();
            maximizeButton = new FontAwesome.Sharp.IconButton();
            closeButton = new FontAwesome.Sharp.IconButton();
            panelchoose = new Panel();
            buttonnewidea = new Button();
            buttonRanking = new Button();
            buttonCompanies = new Button();
            buttonStock = new Button();
            panelforms = new Panel();
            closepanel.SuspendLayout();
            panelchoose.SuspendLayout();
            SuspendLayout();
            // 
            // closepanel
            // 
            closepanel.BackColor = Color.FromArgb(64, 64, 64);
            closepanel.Controls.Add(labeltitle);
            closepanel.Controls.Add(minimizeButton);
            closepanel.Controls.Add(maximizeButton);
            closepanel.Controls.Add(closeButton);
            closepanel.Dock = DockStyle.Top;
            closepanel.Location = new Point(0, 0);
            closepanel.Name = "closepanel";
            closepanel.Size = new Size(1250, 61);
            closepanel.TabIndex = 1;
            closepanel.MouseDown += closepanel_MouseDown;
            closepanel.MouseMove += closepanel_MouseMove;
            closepanel.MouseUp += closepanel_MouseUp;
            // 
            // labeltitle
            // 
            labeltitle.Anchor = AnchorStyles.Left;
            labeltitle.AutoSize = true;
            labeltitle.Font = new Font("Segoe UI", 10F);
            labeltitle.ForeColor = SystemColors.Window;
            labeltitle.Location = new Point(0, 10);
            labeltitle.Name = "labeltitle";
            labeltitle.Size = new Size(293, 37);
            labeltitle.TabIndex = 7;
            labeltitle.Text = "AIR-mazing Predictions";
            // 
            // minimizeButton
            // 
            minimizeButton.Dock = DockStyle.Right;
            minimizeButton.FlatStyle = FlatStyle.Flat;
            minimizeButton.ForeColor = Color.FromArgb(64, 64, 64);
            minimizeButton.IconChar = FontAwesome.Sharp.IconChar.WindowMinimize;
            minimizeButton.IconColor = Color.FromArgb(158, 158, 158);
            minimizeButton.IconFont = FontAwesome.Sharp.IconFont.Auto;
            minimizeButton.IconSize = 35;
            minimizeButton.Location = new Point(1010, 0);
            minimizeButton.Name = "minimizeButton";
            minimizeButton.Size = new Size(80, 61);
            minimizeButton.TabIndex = 6;
            minimizeButton.UseVisualStyleBackColor = true;
            minimizeButton.Click += minimizeButton_Click;
            // 
            // maximizeButton
            // 
            maximizeButton.Dock = DockStyle.Right;
            maximizeButton.FlatStyle = FlatStyle.Flat;
            maximizeButton.ForeColor = Color.FromArgb(64, 64, 64);
            maximizeButton.IconChar = FontAwesome.Sharp.IconChar.WindowMaximize;
            maximizeButton.IconColor = Color.FromArgb(158, 158, 158);
            maximizeButton.IconFont = FontAwesome.Sharp.IconFont.Auto;
            maximizeButton.IconSize = 35;
            maximizeButton.Location = new Point(1090, 0);
            maximizeButton.Name = "maximizeButton";
            maximizeButton.Size = new Size(80, 61);
            maximizeButton.TabIndex = 5;
            maximizeButton.UseVisualStyleBackColor = true;
            maximizeButton.Click += maximizeButton_Click;
            // 
            // closeButton
            // 
            closeButton.Dock = DockStyle.Right;
            closeButton.FlatStyle = FlatStyle.Flat;
            closeButton.ForeColor = Color.FromArgb(64, 64, 64);
            closeButton.IconChar = FontAwesome.Sharp.IconChar.X;
            closeButton.IconColor = Color.FromArgb(158, 158, 158);
            closeButton.IconFont = FontAwesome.Sharp.IconFont.Auto;
            closeButton.IconSize = 35;
            closeButton.Location = new Point(1170, 0);
            closeButton.Name = "closeButton";
            closeButton.Size = new Size(80, 61);
            closeButton.TabIndex = 3;
            closeButton.UseVisualStyleBackColor = true;
            closeButton.Click += closeButton_Click;
            // 
            // panelchoose
            // 
            panelchoose.Controls.Add(buttonnewidea);
            panelchoose.Controls.Add(buttonRanking);
            panelchoose.Controls.Add(buttonCompanies);
            panelchoose.Controls.Add(buttonStock);
            panelchoose.Dock = DockStyle.Top;
            panelchoose.Location = new Point(0, 61);
            panelchoose.Name = "panelchoose";
            panelchoose.Size = new Size(1250, 61);
            panelchoose.TabIndex = 2;
            // 
            // buttonnewidea
            // 
            buttonnewidea.BackColor = Color.FromArgb(64, 64, 64);
            buttonnewidea.Dock = DockStyle.Right;
            buttonnewidea.FlatStyle = FlatStyle.Popup;
            buttonnewidea.ForeColor = SystemColors.Window;
            buttonnewidea.Location = new Point(1019, 0);
            buttonnewidea.Name = "buttonnewidea";
            buttonnewidea.Size = new Size(231, 61);
            buttonnewidea.TabIndex = 3;
            buttonnewidea.Text = "Try a new Idea";
            buttonnewidea.UseVisualStyleBackColor = false;
            buttonnewidea.Click += buttonnewidea_Click;
            // 
            // buttonRanking
            // 
            buttonRanking.BackColor = Color.FromArgb(64, 64, 64);
            buttonRanking.Dock = DockStyle.Left;
            buttonRanking.FlatStyle = FlatStyle.Popup;
            buttonRanking.ForeColor = SystemColors.Window;
            buttonRanking.Location = new Point(462, 0);
            buttonRanking.Name = "buttonRanking";
            buttonRanking.Size = new Size(231, 61);
            buttonRanking.TabIndex = 2;
            buttonRanking.Text = "Stock Ranking";
            buttonRanking.UseVisualStyleBackColor = false;
            buttonRanking.Click += buttonRanking_Click;
            // 
            // buttonCompanies
            // 
            buttonCompanies.BackColor = Color.FromArgb(64, 64, 64);
            buttonCompanies.Dock = DockStyle.Left;
            buttonCompanies.FlatStyle = FlatStyle.Popup;
            buttonCompanies.ForeColor = SystemColors.Window;
            buttonCompanies.Location = new Point(231, 0);
            buttonCompanies.Name = "buttonCompanies";
            buttonCompanies.Size = new Size(231, 61);
            buttonCompanies.TabIndex = 1;
            buttonCompanies.Text = "Similiar Companies";
            buttonCompanies.UseVisualStyleBackColor = false;
            buttonCompanies.Click += buttonCompanies_Click;
            // 
            // buttonStock
            // 
            buttonStock.BackColor = Color.FromArgb(64, 64, 64);
            buttonStock.Dock = DockStyle.Left;
            buttonStock.FlatStyle = FlatStyle.Popup;
            buttonStock.ForeColor = SystemColors.Window;
            buttonStock.Location = new Point(0, 0);
            buttonStock.Name = "buttonStock";
            buttonStock.Size = new Size(231, 61);
            buttonStock.TabIndex = 0;
            buttonStock.Text = "Stock Prediction";
            buttonStock.UseVisualStyleBackColor = false;
            buttonStock.Click += buttonStock_Click;
            // 
            // panelforms
            // 
            panelforms.Dock = DockStyle.Fill;
            panelforms.Location = new Point(0, 122);
            panelforms.Name = "panelforms";
            panelforms.Size = new Size(1250, 547);
            panelforms.TabIndex = 3;
            // 
            // OverviewForm
            // 
            AutoScaleDimensions = new SizeF(13F, 32F);
            AutoScaleMode = AutoScaleMode.Font;
            BackColor = Color.FromArgb(50, 50, 50);
            ClientSize = new Size(1250, 669);
            Controls.Add(panelforms);
            Controls.Add(panelchoose);
            Controls.Add(closepanel);
            FormBorderStyle = FormBorderStyle.None;
            Name = "OverviewForm";
            Text = "OverviewForm";
            closepanel.ResumeLayout(false);
            closepanel.PerformLayout();
            panelchoose.ResumeLayout(false);
            ResumeLayout(false);
        }

        #endregion

        private Panel closepanel;
        private Label labeltitle;
        private FontAwesome.Sharp.IconButton minimizeButton;
        private FontAwesome.Sharp.IconButton maximizeButton;
        private FontAwesome.Sharp.IconButton closeButton;
        private Panel panelchoose;
        private Button buttonStock;
        private Button buttonCompanies;
        private Button buttonRanking;
        private Panel panelforms;
        private Button buttonnewidea;
    }
}