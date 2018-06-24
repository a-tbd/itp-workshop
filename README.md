# ITP Camp Workshop: Machine learning + image making

We'll be using Spell as our remote GPU provider for these demos.
To get Spell go to spell.run and sign up for an account.
Then install using `pip install spell`
Next, login to your account using `spell login` and the login details from the sign up step.

## Transferring Style with [Neural Style](https://github.com/anishathalye/neural-style)

1. Clone the neural style repo
```
git clone https://github.com/anishathalye/neural-style
```

2. Create a styles repo (or clone it from this repo)
```
mkdir styles
# add your styles
```
or
`git clone https://github.com/a-tbd/itp-workshop.git`

3. Upload your styles repo (optional)
The name flag is optional.  Since I have multiple style directories uploaded to Spell, I'll use the name flag to specify that this is my itp_styles directory.
```
spell upload styles [--name itp_styles]
```

You can view your uploaded resources on Spell using `spell ls uploads`

4. Download pretrained model
this url will begin the download: http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat

5. Upload the model to Spell
This isn't strictly necessary, but for some reason the code won't run for me unless I upload it.  This model is also used by other repos so it's handy to have it available online anyway.

The below code will put your file into a directory named vgg.

```
spell upload ~/Downloads/imagenet-vgg-verydeep-19.mat --name vgg
```

To view the file use `spell ls uploads/vgg`

6. Select and upload content images
Create another folder for your content images, add your images, and upload to Spell.

```
mkdir content_images
# add images to your directory
spell upload content_images
```

7. Go into your neural-style dir and run some style transfers!

The flags you'll need for your Spell command include:
- `-t K80/V100` : this indicates what type of GPU you're using.  The K80 is slower and cheaper ($1/hr) and the V100 is faster but more expensive ($4/hr).  The style transfer should only take a few minutes to complete.  If you want to run on CPU, just leave out the -t flag.
- `-m uploads/itp_styles:styles` : you'll need to "mount" your directories so your command can "see" them.  Think of everything that's inside the quotes as your command.  This command is running in a fresh environment, and it isn't able to see the directories you've uploaded to Spell unless you mount them first.  When mounting, you can give your mounted directory a new name if that's easier to remember.  Everything ahead of the colon is your directory as it exists on Spell, everything after the colon is your new name for that directory.  In this case I'm mounting my `uploads/itp_styles` directory and calling it `styles` for this command.
- `-m uploads/content_images:images` : here I'm mounting my content images and calling the directory `images` for this command.
- `-m uploads/vgg:vgg` : lastly, I'm mounting the pre-trained model we downloaded.
- the command: you can see all the different available flags for your command by using `spell run "python neural_style.py --help"` or by checking [here](https://github.com/a-tbd/itp-workshop/neural_style_parameters.md).  In this command, I'm just using the default parameters.

```
cd neural-style

spell run -t K80 -m uploads/itp_styles:styles -m uploads/content_images:images -m uploads/vgg:vgg \
"python neural_style.py \
--network vgg/imagenet-vgg-verydeep-19.mat \
--content images/my_image.jpg \
--styles styles/galaxy.jpg \
--output first_transfer.jpg"

💫 Casting spell #2338…

```

8. Download your results

Next, I'll download my results to a new folder called outputs.  Here, you'll want to replace `<run id>` with your run id, which you can find either at the beginning or end of your run logs, or by checking `spell ps`

```
spell cp runs/<run id> [../outputs]
```

9. Now trying playing around with parameters!

Try changing the default parameters such as `--content-weight` or `--style-layer-weight-exp` to see how this changes your results.

## Training a neural net with [pix2pix](https://github.com/affinelayer/pix2pix-tensorflow.git)