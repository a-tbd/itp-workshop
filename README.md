# ITP Camp Workshop: Machine learning + image making

We'll be using Spell as our remote GPU provider for these demos.
To get Spell go to [spell.run](spell.run) and sign up for an account.
Then install using `pip install spell`
Next, login to your account using `spell login` and the login details from the sign up step.

## Transferring Style with [Neural Style](https://github.com/anishathalye/neural-style)

1. Clone the neural style repo
```
git clone https://github.com/anishathalye/neural-style
```

2. Create a styles repo (or clone it from this repo)
```
$nmkdir styles
# add your styles
```
or
`git clone https://github.com/a-tbd/itp-workshop.git`

3. Upload your styles repo (optional)

The name flag is optional.  Since I have multiple style directories uploaded to Spell, I'll use the name flag to specify that this is my itp_styles directory.
```
$ spell upload styles [--name itp_styles]
```

You can view your uploaded resources on Spell using `spell ls uploads`

4. Download pretrained model

this url will begin the download: http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat

5. Upload the model to Spell
This isn't strictly necessary, but for some reason the code won't run for me unless I upload it.  This model is also used by other repos so it's handy to have it available online anyway.

The below code will put your file into a directory named vgg.

```
$ spell upload ~/Downloads/imagenet-vgg-verydeep-19.mat --name vgg
```

To view the file use `spell ls uploads/vgg`

6. Select and upload content images

Create another folder for your content images, add your images, and upload to Spell.

```
$ mkdir content_images
# add images to your directory
$ spell upload content_images
```

7. Go into your neural-style dir and run some style transfers!

The flags you'll need for your Spell command include:
- `-t K80/V100` : this indicates what type of GPU you're using.  The K80 is slower and cheaper ($1/hr) and the V100 is faster but more expensive ($4/hr).  The style transfer should only take a few minutes to complete.  This code doesn't work on CPU so you'll need to select one of the GPU options.
- `-m uploads/itp_styles:styles` : you'll need to "mount" your directories so your command can "see" them.  Think of everything that's inside the quotes as your command.  This command is running in a fresh environment, and it isn't able to see the directories you've uploaded to Spell unless you mount them first.  When mounting, you can give your mounted directory a new name if that's easier to remember.  Everything ahead of the colon is your directory as it exists on Spell, everything after the colon is your new name for that directory.  In this case I'm mounting my `uploads/itp_styles` directory and calling it `styles` for this command.
- `-m uploads/content_images:images` : here I'm mounting my content images and calling the directory `images` for this command.
- `-m uploads/vgg:vgg` : lastly, I'm mounting the pre-trained model we downloaded.
- the command: you can see all the different available flags for your command by using `spell run "python neural_style.py --help"` or by checking [here](https://github.com/a-tbd/itp-workshop/neural_style_parameters.md).  In this command, I'm just using the default parameters.

```
$ cd neural-style

$ spell run -t K80 -m uploads/itp_styles:styles -m uploads/content_images:images -m uploads/vgg:vgg \
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
$ spell cp runs/<run id> [../outputs]
```

9. Now trying playing around with parameters!

Try changing the default parameters such as `--content-weight` or `--style-layer-weight-exp` to see how this changes your results.

Examples:

Starry night with [tiled moons](https://github.com/a-tbd/itp-workshop/blob/master/styles/many_moons_smaller.jpg) as the style and the default parameters

<img src="https://github.com/a-tbd/itp-workshop/blob/master/imgs/neural_style/starry_moons_default.jpg" alt="Starry night and tiled moon style transfer" width="500px"/>

Starry night with [tiled moons](https://github.com/a-tbd/itp-workshop/blob/master/styles/many_moons_smaller.jpg) as the style and `--style-layer-weight-exp` set to .4

<img src="https://github.com/a-tbd/itp-workshop/blob/master/imgs/neural_style/starry_moons_point4.jpg" alt="Starry Night + Many Moons --style-layer-weight-exp .4" width="500px"/>

Starry night with [tiled moons](https://github.com/a-tbd/itp-workshop/blob/master/styles/many_moons_smaller.jpg) as the style and `--style-layer-weight-exp` set to .2

<img src="https://github.com/a-tbd/itp-workshop/blob/master/imgs/neural_style/starry_moons_point2.jpg" alt="Starry Night + Many Moons --style-layer-weight-exp .2" width="500px"/>


## Training a neural net with [pix2pix](https://github.com/affinelayer/pix2pix-tensorflow.git)

Next we'll take a look at pix2pix, which is a popular repo for artists.

1. clone pix2pix

```
$ git clone https://github.com/affinelayer/pix2pix-tensorflow.git
$ cd pix2pix-tensorflow
```

2. find a video
for this example, we'll going to train a neural net to generate the next frame of a video.  Here I'm using a short video of a cloud, which you can download from my google drive [here](https://drive.google.com/drive/folders/1nmKfQb24ICd9cm9lslre6kHPQqy3KB3H?usp=sharing).
```
$ mkdir video_input 
# Add a video to your new directory
$ spell upload video_input
```

3. get frames from the video

In order to train our neural net, we'll need to get the frames from the video, and pair each frame with the frame that comes after.  Then we can train the neural net to generate the next frame of the video, based on a frame we're inputting.

This code snippet will output a series of numbered pngs which are each frame of the video.
- `-m uploads/video_input:video` : this part of the command mounts the directory which we just uploaded that contains our video.  This makes the video we uploaded available to our command.
- `--apt ffmpeg` : In order to create the video frames, we'll need to use ([ffmpeg](https://www.ffmpeg.org/)).  Spell doesn't come pre-installed with ffmpeg, so we'll need to include the package in our run using `--apt`.  
- `ffmpeg -i video/face.mp4 -r 1/1 $filename%04d.png` : this command is looping through all the frames in our video `fast_clouds.mp4` and then saving that frame with the name 0001.png, 0002.png etc.  The `%4d` indicates that we want to give our numbers 4 digit places.  If you have a lot of frames you may need to change this to `$05d`.

```
$ spell run -m uploads/video_input:video --apt ffmpeg "ffmpeg -i video/fast_clouds.mp4 -r 1/1 $filename%04d.png"
💫 Casting spell #123…
```

4. Next we'll download our frames

You'll need to replace `<run id>` with the id from the previous run. You can find this at the beginning or end of your run logs, or using `spell ps`
```
$ spell cp runs/<run id> fast_clouds_frames
```

5. Rename your files to pair them

In order to train our neural net using pix2pix, we'll need to create paired images.  To do this, we need two directories with images that have identical names so the pix2pix script can create pairs.  Since we want to pair each frame of one directory with the next frame, we'll create a second directory and rename the files so they are shifted by one.

```
# create a second directory that is a copy of all the frames in your original directory
$ cp -r fast_clouds_frames fast_clouds_frames_2 

# rename them shifted by one.  This needs to be run from within the directory that contains your images
$ cd fast_clouds_frames_2
$ a=0; for i in *.png; do new=$(printf "%04d.png" "$a"); mv -i -- "$i" "$new"; let a=a+1; done 
```
You'll need to remove 0000.png from the new dir, and the last file from your old dir to make sure the numbers of the files all match up.

6. Upload your images

```
$ mkdir paired_images
$ mv fast_clouds_frames paired_images
$ mv fast_clouds_frames_2 paired_images
$ spell upload paired_images
```

7. resize the images

The pix2pix repo comes with a script for resizing your images.  Remember the run ids for each of these processes.
```
$ spell run -m uploads/paired_images:images -t K80 "python tools/process.py --input_dir images/face_frames_out --operation resize --output_dir resized"
💫 Casting spell #100…

# Repeat the resizing process for your other directory
$ spell run -m uploads/paired_images:images -t K80 "python tools/process.py --input_dir images/face_frames_out_2 --operation resize --output_dir resized"
💫 Casting spell #101…
```

8. combine your images

Pix2pix needs your images to be in pairs, and it comes with a script for creating the paired images. You'll need the run ids from your resizing image step for this.
- `-m runs/100:a` : mount your resized images from your first run
- `-m runs/101:b` : mount your resized images from your second run
```
$ spell run -m runs/100:a -m runs/101:b "python tools/process.py --input_dir a/resized --b_dir b/resized --operation combine --output_dir combined"
💫 Casting spell #102…
```

9. train!

Now you're ready to train your neural net on your paired images.  You'll need to:
- `-t K80` : you'll want a GPU for the training step.  You can select either the `K80` or the `V100`.  This code doesn't work on CPU :(
- `-m runs/102:images_train` : You'll need to mount your directory of paired images from the previous run.  If you forgot the run id, you can check it using `spell ps`
- `--python2` : although pix2pix should work with python 3, it wasn't working for me during testing.  Using the `--python2` flag to change the environment to python 2 did the trick.

```
$ spell run -t K80 -m runs/102:images_train --python2 "python pix2pix.py --mode train --output_dir face_train --max_epochs 200 --input_dir images_train/combined --which_direction BtoA"
💫 Casting spell #103…
```

10. Test!

This step can also be used to generate new images.  Here, I'm mounting the same directory we used for training in order to test, but you could use a new image, or new directory of images.

- `-t K80` : you can run this step on a CPU (which I've done below by leaving out the `-t` flag).  You can also specify a GPU if you'd like.
- `-m runs/102/combined:images` : we're mounting the same directory we created in step 8 and mounted in step 9.
- `-m runs/103/clouds_train:clouds_train` : here, we're mounting our trained neural net which was created during the training step.  if you don't remember the run id from that step, you can find it using `spell ps`  

```
$ spell run -m runs/102/combined:images --python2 -m runs/103/clouds_train:clouds_train "python pix2pix.py --mode test --output_dir clouds_test --input_dir images --checkpoint clouds_train"
💫 Casting spell #104…
```

Finally, we'll download the output of our test step to see our new, generated images:
```
$ spell cp runs/104 ../output_clouds
```

In this example, we generated the next frame based on one frame.  In order to create a "dream" video like the [Mario Klingemann fireworks](https://www.fastcodesign.com/90156087/an-ai-learned-to-make-fireworks-and-theyre-mesmerizing) example, you'll need to use one of your generated frames as the input image when you run the `--test` step and keep doing that to generate more frames.

In other words, instead of running *all* your images through the `--test` script, just run one image.  Then take that output and use that as your next input image.  And keep repeating.  

Try playing around with different pairs of images, changing the `--max_epochs` flag, or the `--which_direction` flag to create new iamges and artworks.

Input image (frame from the [cloud video](https://github.com/a-tbd/itp-workshop/tree/master/cloud_frames))

<img src="https://github.com/a-tbd/itp-workshop/blob/master/imgs/pix2pix/0001-inputs.png" alt="Input image, frame from cloud video" width="300px"/>

Output image (next frame as generated by the neural network we trained)

<img src="https://github.com/a-tbd/itp-workshop/blob/master/imgs/pix2pix/0001-outputs.png" alt="Output image, frame from cloud video" width="300px"/>

Target image (actual next frame of the cloud video)

<img src="https://github.com/a-tbd/itp-workshop/blob/master/imgs/pix2pix/0001-targets.png" alt="Target image, actual next frame of cloud video" width="300px"/>


## Other repos/resources to look at

1. https://github.com/cysmith/neural-style-tf
2. https://github.com/ProGamerGov/Neural-Zoom
3. https://github.com/lengstrom/fast-style-transfer

There's a tutorial for using fast style transfer with Spell [here](https://spell.run/docs/transferring_style/)

4. Gene Kogan's [machine learning for artists](https://ml4a.github.io/guides/) website and [slack channel](https://ml-4a.slack.com/join/shared_invite/enQtMjcxMjUwNDQ0NDAzLWVmMTFmMTc3ZDJlYTExNGE1ZmQ4NTk1NGJhZWI3ODJmYWI2ZjgxYmQ3NWMzOWU3NTE3ZDYxOTBkMDg5Nzg5M2I)
5. Info on doing hi res style transfer : https://github.com/jcjohnson/neural-style/issues/351
