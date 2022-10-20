using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Timers;
using System.Linq;
using System.Threading.Tasks;
using LedCSharp;

namespace WindowsFormsApp1
{

    public partial class Form2 : Form
    {
        
        System.Timers.Timer t;

        static keyboardNames get_keyname(char key)
        {
            if (key == 'a' || key == 'A')
                return keyboardNames.A;
            if (key == 'b' || key == 'B')
                return keyboardNames.B;
            if (key == 'c' || key == 'C')
                return keyboardNames.C;
            if (key == 'd' || key == 'D')
                return keyboardNames.D;
            if (key == 'e' || key == 'E')
                return keyboardNames.E;
            if (key == 'f' || key == 'F')
                return keyboardNames.F;
            if (key == 'g' || key == 'G')
                return keyboardNames.G;
            if (key == 'h' || key == 'H')
                return keyboardNames.H;
            if (key == 'i' || key == 'I')
                return keyboardNames.I;
            if (key == 'j' || key == 'J')
                return keyboardNames.J;
            if (key == 'k' || key == 'K')
                return keyboardNames.K;
            if (key == 'l' || key == 'L')
                return keyboardNames.L;
            if (key == 'm' || key == 'M')
                return keyboardNames.M;
            if (key == 'n' || key == 'N')
                return keyboardNames.N;
            if (key == 'o' || key == 'O')
                return keyboardNames.O;
            if (key == 'p' || key == 'P')
                return keyboardNames.P;
            if (key == 'q' || key == 'Q')
                return keyboardNames.Q;
            if (key == 'r' || key == 'R')
                return keyboardNames.R;
            if (key == 's' || key == 'S')
                return keyboardNames.S;
            if (key == 't' || key == 'T')
                return keyboardNames.T;
            if (key == 'u' || key == 'U')
                return keyboardNames.U;
            if (key == 'v' || key == 'V')
                return keyboardNames.V;
            if (key == 'w' || key == 'W')
                return keyboardNames.W;
            if (key == 'x' || key == 'X')
                return keyboardNames.X;
            if (key == 'y' || key == 'Y')
                return keyboardNames.Y;
            if (key == 'z' || key == 'Z')
                return keyboardNames.Z;
            if (key == ' ')
                return keyboardNames.SPACE;
            if (key == Convert.ToChar(13))
                return keyboardNames.ENTER;
            if (key == Convert.ToChar(9))
                return keyboardNames.TAB;
            if (key == ',')
                return keyboardNames.COMMA;
            if (key == '.')
                return keyboardNames.PERIOD;
            if (key == Convert.ToChar(39))
                return keyboardNames.APOSTROPHE;
            if (key == ';')
                return keyboardNames.SEMICOLON;
            if (key == ':')
                return keyboardNames.SEMICOLON;
            if (key == '"')
                return keyboardNames.APOSTROPHE;
            if (key == '!')
                return keyboardNames.ONE;
            if (key == '?')
                return keyboardNames.FORWARD_SLASH;
            if (key == '-')
                return keyboardNames.MINUS;
            else return 0;
        }

        public Form2()
        {
            InitializeComponent();
            label1.Text = ("Book " + (Program.book + 1));
            label2.Text = ("Time Limit: " + (Program.timespan) + " min");

            if (Program.book == 0)
            {
                Program.lines = File.ReadAllText(@"..\..\..\..\Resource\T1.txt");
            }
            else if (Program.book == 1)
            {
                Program.lines = File.ReadAllText(@"..\..\..\..\Resource\T2.txt");
            }
            else 
            {
                Program.lines = File.ReadAllText(@"..\..\..\..\Resource\T3.txt");
            }
            

            // Initialize the LED SDK
            bool LedInitialized = LogitechGSDK.LogiLedInitWithName("SetTargetZone Sample C#");


            LogitechGSDK.LogiLedSetTargetDevice(LogitechGSDK.LOGI_DEVICETYPE_ALL);

            // Set all devices to Black
            LogitechGSDK.LogiLedSetLighting(0, 0, 0);
            //string lines = File.ReadAllText(@"C:\Users\dppss\OneDrive\桌面\Harunoyu\txtsources\T1.txt");

            Char[] mychars = { ' ', ',', '.', '?', '!', '"', '\'', '\r', '\n', '\t' };

            

            string tmp = "";
            for (int i = 0; i < Program.lines.Length; i++)
            {
                if (i != 0 && mychars.Contains(Program.lines[i]))
                {
                    if (tmp != "") Program.splited_str.Add(tmp);
                    tmp = "";
                    Program.splited_str.Add("" + Program.lines[i]);
                }
                else if (i == 0 && mychars.Contains(Program.lines[i]))
                {
                    Program.splited_str.Add("" + Program.lines[i]);
                }
                else tmp += Program.lines[i];
            }
            Program.splited_str.Add(tmp);
            for (int i = Program.cur_n_in_str; i < Program.splited_str[Program.cur_string].Length; i++)
            {
                LogitechGSDK.LogiLedSetLightingForKeyWithKeyName(get_keyname(Program.splited_str[Program.cur_string][i]), 0, 45, 45);
            }
            LogitechGSDK.LogiLedSetLightingForKeyWithKeyName(get_keyname(Program.ans), 100, 80, 0);
            if (Program.ans == '!' || Program.ans == '?' || Program.ans == '"' || Program.ans == ':')
            {
                LogitechGSDK.LogiLedSetLightingForKeyWithKeyName(keyboardNames.RIGHT_SHIFT, 0, 100, 100);
                LogitechGSDK.LogiLedSetLightingForKeyWithKeyName(keyboardNames.LEFT_SHIFT, 0, 100, 100);
            }
        }

        private void Form2_Load(object sender, EventArgs e)
        {
            t = new System.Timers.Timer();
            t.Interval = 1000;
            t.Elapsed += OnTimeEvent;
        }

        private void OnTimeEvent(object sender, ElapsedEventArgs e)
        {
            Invoke(new Action(() =>
           {
               Program.s += 1;
               if (Program.s == 60)
               {
                   Program.s = 0;
                   Program.m += 1;
               }
               if (Program.m == 60)
               {
                   Program.m = 0;
                   Program.h += 1;
               }
               textBox1.Text = String.Format("{0}:{1}:{2}", Program.h.ToString().PadLeft(2, '0'), Program.m.ToString().PadLeft(2, '0'), Program.s.ToString().PadLeft(2, '0'));
               if (Program.m == Program.timespan)
               {
                   t.Stop();
                   Form3 frame = new Form3();
                   frame.Show();
                   this.Close();
               }
           }));
            
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void textBox1_TextChanged_1(object sender, EventArgs e)
        {

        }

        private void button3_Click(object sender, EventArgs e)
        {
            t.Stop();
        }

        private void label1_Click_1(object sender, EventArgs e)
        {
            
        }

        private void label2_Click(object sender, EventArgs e)
        {

        }

        private void label3_Click(object sender, EventArgs e)
        {

        }

        private void richTextBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void richTextBox2_KeyPress(object sender, KeyPressEventArgs e)
        {
            
            //string lines = "abcde";



            //MessageBox.Show("" + ans);




            if (Program.corret_n < Program.lines.Length)
            {

                //textBox2.Text = Convert.ToString(Program.ans);

                if (e.KeyChar == Program.ans)
                {
                   
                    LogitechGSDK.LogiLedSetLighting(0, 0, 0);
                    LogitechGSDK.LogiLedPulseSingleKey(keyboardNames.G_LOGO, 0, 0, 100, 100, 100, 0, 5000, true);
                    //Console.WriteLine(e.KeyChar);
                    


                    if (Program.cur_n_in_str < Program.splited_str[Program.cur_string].Length - 1)
                    {
                        Program.cur_n_in_str++;
                    }
                    else
                    {
                        Program.cur_n_in_str = 0;
                        Program.cur_string++;
                        //if (Program.cur_string >= Program.splited_str.Count())
                            //break;
                            if (Program.splited_str[Program.cur_string] == "\n")
                            {
                                Program.cur_string++;
                            }
                    }

                    if (e.KeyChar == Convert.ToChar(13))
                        Program.corret_n++;

                    Program.corret_n++;
                    label3.Text = ("Correct: " + Program.corret_n);
                    //break;
                }
                else
                {
                    if(e.KeyChar != Convert.ToChar(8))
                        Program.error_n++;
                    label4.Text = ("Wrong: " + Program.error_n);
                    LogitechGSDK.LogiLedSetLightingForKeyWithKeyName(get_keyname(e.KeyChar), 100, 0, 0);
                    //break;
                }
                Program.ans = Program.lines[Program.corret_n];
                textBox2.Text = Convert.ToString(Program.ans);
                for (int i = Program.cur_n_in_str; i < Program.splited_str[Program.cur_string].Length; i++)
                {
                    LogitechGSDK.LogiLedSetLightingForKeyWithKeyName(get_keyname(Program.splited_str[Program.cur_string][i]), 0, 45, 45);
                }
                LogitechGSDK.LogiLedSetLightingForKeyWithKeyName(get_keyname(Program.ans), 100, 80, 0);
                if (Program.ans == '!' || Program.ans == '?' || Program.ans == '"' || Program.ans == ':')
                {
                    LogitechGSDK.LogiLedSetLightingForKeyWithKeyName(keyboardNames.RIGHT_SHIFT, 0, 100, 100);
                    LogitechGSDK.LogiLedSetLightingForKeyWithKeyName(keyboardNames.LEFT_SHIFT, 0, 100, 100);
                }
            }
            else
            {
                t.Stop();
                Form3 frame = new Form3();
                frame.Show();
                this.Close();
            }
        }

        
        private void richTextBox2_KeyDown(object sender, KeyEventArgs e)
        {
            
        }

        private void richTextBox2_TextChanged(object sender, EventArgs e)
        {

        }

        private void textBox2_TextChanged(object sender, EventArgs e)
        {

        }

        private void label4_Click(object sender, EventArgs e)
        {

        }

        private void button2_Click(object sender, EventArgs e)
        {
            t.Stop();
            Form3 frame = new Form3();
            frame.Show();
            this.Close();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            //TextReader reader = new StreamReader(@"C:\Users\dppss\OneDrive\桌面\Harunoyu\txtsources\T1.txt");
            richTextBox1.Text = Program.lines;
            //reader.Close();
            t.Start();
            button1.Text = ("Contunue");
        }

    }
}
