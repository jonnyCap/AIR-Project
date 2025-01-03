using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows.Forms;

namespace AIR
{
    public partial class SimilarCompaniesForm : Form
    {
        public SimilarCompaniesForm()
        {
            InitializeComponent();
        }

        public void setListinBox(string ticker, string similiarity)
        {
            
            
            listofcompanies.Items.Add(ticker);
            

            //foreach (var item in similiarity_list_)
            //{
            //    listofcompanies.Items.Add(item);
            //}
        }
        public void sizeOfCompanies(int x, int y)
        {
            listofcompanies.Size = new Size(x, y);
        }
        public void changeFontSize(int size)
        {
            listofcompanies.Font = new Font("Segoe UI", size, FontStyle.Regular);
        }
        public void clearListBox()
        {
            listofcompanies.Items.Clear();
        }
        private void listofcompanies_SelectedIndexChanged(object sender, EventArgs e)
        {

        }
    }
}
