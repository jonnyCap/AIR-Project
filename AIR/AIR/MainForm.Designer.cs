namespace AIR
{
    partial class MainForm
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
            fileSystemWatcher1 = new FileSystemWatcher();
            closepanel = new Panel();
            labeltitle = new Label();
            minimizeButton = new FontAwesome.Sharp.IconButton();
            maximizeButton = new FontAwesome.Sharp.IconButton();
            closeButton = new FontAwesome.Sharp.IconButton();
            tableLayoutPanel = new TableLayoutPanel();
            label1 = new Label();
            iconButtonAnalyze = new FontAwesome.Sharp.IconButton();
            paneltextbox = new Panel();
            textBoxidea = new TextBox();
            progressBar = new ProgressBar();
            ((System.ComponentModel.ISupportInitialize)fileSystemWatcher1).BeginInit();
            closepanel.SuspendLayout();
            tableLayoutPanel.SuspendLayout();
            paneltextbox.SuspendLayout();
            SuspendLayout();
            // 
            // fileSystemWatcher1
            // 
            fileSystemWatcher1.EnableRaisingEvents = true;
            fileSystemWatcher1.SynchronizingObject = this;
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
            closepanel.TabIndex = 0;
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
            labeltitle.Click += labeltitle_Click;
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
            // tableLayoutPanel
            // 
            tableLayoutPanel.ColumnCount = 3;
            tableLayoutPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.33333F));
            tableLayoutPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.3333359F));
            tableLayoutPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.3333359F));
            tableLayoutPanel.Controls.Add(label1, 0, 0);
            tableLayoutPanel.Controls.Add(iconButtonAnalyze, 1, 2);
            tableLayoutPanel.Controls.Add(paneltextbox, 0, 1);
            tableLayoutPanel.Controls.Add(progressBar, 0, 3);
            tableLayoutPanel.Dock = DockStyle.Fill;
            tableLayoutPanel.Location = new Point(0, 61);
            tableLayoutPanel.Name = "tableLayoutPanel";
            tableLayoutPanel.RowCount = 4;
            tableLayoutPanel.RowStyles.Add(new RowStyle(SizeType.Percent, 13.636364F));
            tableLayoutPanel.RowStyles.Add(new RowStyle(SizeType.Percent, 45.4545441F));
            tableLayoutPanel.RowStyles.Add(new RowStyle(SizeType.Percent, 20.454546F));
            tableLayoutPanel.RowStyles.Add(new RowStyle(SizeType.Percent, 20.454546F));
            tableLayoutPanel.RowStyles.Add(new RowStyle(SizeType.Absolute, 20F));
            tableLayoutPanel.RowStyles.Add(new RowStyle(SizeType.Absolute, 20F));
            tableLayoutPanel.Size = new Size(1250, 669);
            tableLayoutPanel.TabIndex = 1;
            // 
            // label1
            // 
            label1.Anchor = AnchorStyles.Bottom | AnchorStyles.Left;
            label1.AutoSize = true;
            label1.Font = new Font("Segoe UI", 10F);
            label1.ForeColor = SystemColors.Window;
            label1.Location = new Point(3, 39);
            label1.Margin = new Padding(3, 0, 10, 15);
            label1.Name = "label1";
            label1.Size = new Size(308, 37);
            label1.TabIndex = 8;
            label1.Text = "Enter Your Business Idea:";
            label1.Click += label1_Click;
            // 
            // iconButtonAnalyze
            // 
            iconButtonAnalyze.Anchor = AnchorStyles.None;
            iconButtonAnalyze.FlatStyle = FlatStyle.Flat;
            iconButtonAnalyze.ForeColor = SystemColors.Window;
            iconButtonAnalyze.IconChar = FontAwesome.Sharp.IconChar.None;
            iconButtonAnalyze.IconColor = Color.FromArgb(50, 50, 50);
            iconButtonAnalyze.IconFont = FontAwesome.Sharp.IconFont.Auto;
            iconButtonAnalyze.Location = new Point(419, 446);
            iconButtonAnalyze.Margin = new Padding(3, 15, 3, 3);
            iconButtonAnalyze.Name = "iconButtonAnalyze";
            iconButtonAnalyze.Size = new Size(410, 46);
            iconButtonAnalyze.TabIndex = 9;
            iconButtonAnalyze.Text = "Analyze Idea";
            iconButtonAnalyze.UseVisualStyleBackColor = true;
            iconButtonAnalyze.Click += iconButtonAnalyze_Click;
            // 
            // paneltextbox
            // 
            tableLayoutPanel.SetColumnSpan(paneltextbox, 3);
            paneltextbox.Controls.Add(textBoxidea);
            paneltextbox.Dock = DockStyle.Fill;
            paneltextbox.Location = new Point(30, 94);
            paneltextbox.Margin = new Padding(30, 3, 30, 3);
            paneltextbox.Name = "paneltextbox";
            paneltextbox.Size = new Size(1190, 298);
            paneltextbox.TabIndex = 10;
            // 
            // textBoxidea
            // 
            textBoxidea.BackColor = Color.FromArgb(64, 64, 64);
            textBoxidea.BorderStyle = BorderStyle.None;
            textBoxidea.Dock = DockStyle.Fill;
            textBoxidea.ForeColor = SystemColors.Window;
            textBoxidea.Location = new Point(0, 0);
            textBoxidea.Multiline = true;
            textBoxidea.Name = "textBoxidea";
            textBoxidea.Size = new Size(1190, 298);
            textBoxidea.TabIndex = 1;
            // 
            // progressBar
            // 
            tableLayoutPanel.SetColumnSpan(progressBar, 3);
            progressBar.Dock = DockStyle.Top;
            progressBar.Location = new Point(30, 534);
            progressBar.Margin = new Padding(30, 3, 30, 3);
            progressBar.Name = "progressBar";
            progressBar.Size = new Size(1190, 50);
            progressBar.TabIndex = 11;
            // 
            // MainForm
            // 
            AutoScaleMode = AutoScaleMode.None;
            BackColor = Color.FromArgb(50, 50, 50);
            ClientSize = new Size(1250, 730);
            Controls.Add(tableLayoutPanel);
            Controls.Add(closepanel);
            ForeColor = SystemColors.ActiveCaptionText;
            FormBorderStyle = FormBorderStyle.None;
            Name = "MainForm";
            Text = "MainForm";
            ((System.ComponentModel.ISupportInitialize)fileSystemWatcher1).EndInit();
            closepanel.ResumeLayout(false);
            closepanel.PerformLayout();
            tableLayoutPanel.ResumeLayout(false);
            tableLayoutPanel.PerformLayout();
            paneltextbox.ResumeLayout(false);
            paneltextbox.PerformLayout();
            ResumeLayout(false);
        }

        #endregion

        private FileSystemWatcher fileSystemWatcher1;
        private Panel closepanel;
        private FontAwesome.Sharp.IconButton maximizeButton;
        private FontAwesome.Sharp.IconButton closeButton;
        private FontAwesome.Sharp.IconButton minimizeButton;
        private Label labeltitle;
        private TableLayoutPanel tableLayoutPanel;
        private Label label1;
        private FontAwesome.Sharp.IconButton iconButtonAnalyze;
        private Panel paneltextbox;
        private TextBox textBoxidea;
        private ProgressBar progressBar;
    }
}